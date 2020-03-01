# import basic module
import logging, numpy as np

# import deeplearning module
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from copy import deepcopy

from model.unet_model_relu_ms import UNet_Relu_MS
from utils.contour_loss import Contour_loss
from utils import pytorch_msssim

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, writer):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.writer = writer
        self.epsilon = 1e-10

        self.net = UNet_Relu_MS(n_channels=args.imgc, n_classes=args.output_channel)
        logging.info(f'Network:\n'
                 f'\t{self.net.n_channels} input channels\n'
                 f'\t{self.net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if self.net.bilinear else "Dilated conv"} upscaling')
        
        if args.load:
            net.load_state_dict(
                torch.load(args.load, map_location=device)
            )
            logging.info(f'Model loaded from {args.load}')
        
        # define loss
        self.mse_loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
            
        # define optimizer for meta learning
        if args.optimizer == "adam" :
            self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        elif args.optimizer == "rmsprop":
            self.meta_optim = optim.RMSprop(net.parameters(), lr=self.meta_lr, weight_decay=self.weight_decay)
        else :
            raise ValueError("Wrong Optimzer !")


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, sun_imgs_x_spt, sun_imgs_y_spt, sun_imgs_x_qry, sun_imgs_y_qry, step):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        fastweight_net = copy.deepcopy(self.net)
        
        task_num, setsz, c_, h, w = sun_imgs_x_spt.size()
        querysz = sun_imgs_x_qry.size(1)

        loss_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        
        # set epsilon
        epsilon=1e-10
        alpha = 0.12
        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            reflection_pred = self.net(sun_imgs_x_spt[i])
            reflection_pred = reflection_pred + epsilon
            reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
            
            restoration_imgs_pred = torch.zeros(*sun_imgs_x_spt[i].shape)
            restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_spt[:,0,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_spt[:,1,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_spt[:,2,:,:], reflection_pred[:,0,:,:])
            
            contour_loss = Contour_loss(K=5)
            weight_map = contour_loss.forward(sun_imgs_y_spt[i]).cpu()
            
            restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
            gt_imgs = torch.mul(sun_imgs_y_spt[i], weight_map)
            
            mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
            
            ms_ssim_loss_fc = pytorch_msssim.MSSSIM(window_size = 7)
            ms_ssim_loss = ms_ssim_loss_fc(restoration_imgs_pred, gt_imgs)
            
            loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ms_ssim_loss)) #+ REGULARIZATION * reg_loss

            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                
                # 1. run the i-th task and compute loss for k=0
                reflection_pred = self.net(sun_imgs_x_qry[i], vars=None)
                reflection_pred = reflection_pred + epsilon
                reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item

                restoration_imgs_pred = torch.zeros(*sun_imgs_x_qry[i].shape)
                restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_qry[:,0,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_qry[:,1,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_qry[:,2,:,:], reflection_pred[:,0,:,:])

                contour_loss = Contour_loss(K=5)
                weight_map = contour_loss.forward(sun_imgs_y_spt[i]).cpu()

                restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                gt_imgs = torch.mul(sun_imgs_y_spt[i], weight_map)

                mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
                
                ms_ssim_loss_fc = pytorch_msssim.MSSSIM(window_size = 7)
                ms_ssim_loss = ms_ssim_loss_fc(restoration_imgs_pred, gt_imgs)

                loss_q[0] = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ms_ssim_loss)) #+ REGULARIZATION * reg_loss
                
                # writer image
                self.writer.add_images('masks/true', gt_imgs, step)
                self.writer.add_images('masks/pred', restoration_imgs_pred, step)
                
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                
                # 1. run the i-th task and compute loss for k=0
                # not use original net, use copy one
                fastweight_param = list(fastweight_net.parameters())
                for idx in range(0, len(fastweight_param)):
                    fastweight_param[idx].data[:] = fast_weights[idx].data[:]
                reflection_pred = fastweight_net(sun_imgs_x_qry[i])
                reflection_pred = reflection_pred + epsilon
                reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item

                restoration_imgs_pred = torch.zeros(*sun_imgs_x_qry[i].shape)
                restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_qry[:,0,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_qry[:,1,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_qry[:,2,:,:], reflection_pred[:,0,:,:])

                contour_loss = Contour_loss(K=5)
                weight_map = contour_loss.forward(sun_imgs_y_qry[i]).cpu()

                restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                gt_imgs = torch.mul(sun_imgs_y_qry[i], weight_map)

                mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
                
                ms_ssim_loss_fc = pytorch_msssim.MSSSIM(window_size = 7)
                ms_ssim_loss = ms_ssim_loss_fc(restoration_imgs_pred, gt_imgs)

                loss_q[1] = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ms_ssim_loss)) #+ REGULARIZATION * reg_loss
                
                # writer image
                self.writer.add_images('masks/true', gt_imgs, step)
                self.writer.add_images('masks/pred', restoration_imgs_pred, step)
                
            for k in range(1, self.update_step):
                
                # 1. run the i-th task and compute loss for k=0
                reflection_pred = self.net(sun_imgs_x_spt[i])
                reflection_pred = reflection_pred + epsilon
                reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item

                restoration_imgs_pred = torch.zeros(*sun_imgs_x_spt[i].shape)
                restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_spt[:,0,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_spt[:,1,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_spt[:,2,:,:], reflection_pred[:,0,:,:])

                contour_loss = Contour_loss(K=5)
                weight_map = contour_loss.forward(sun_imgs_y_spt[i]).cpu()

                restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                gt_imgs = torch.mul(sun_imgs_y_spt[i], weight_map)
                
                mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
                
                ms_ssim_loss_fc = pytorch_msssim.MSSSIM(window_size = 7)
                ms_ssim_loss = ms_ssim_loss_fc(restoration_imgs_pred, gt_imgs)

                loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ms_ssim_loss)) #+ REGULARIZATION * reg_loss
                
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                
                # 1. run the i-th task and compute loss for k=0
                # not use original net, use copy one
                fastweight_param = list(fastweight_net.parameters()
                for idx in range(0, len(fastweight_param)):
                    fastweight_param[idx].data[:] = fast_weights[idx].data[:]
                reflection_pred = fastweight_net(sun_imgs_x_qry[i], fast_weights)
                reflection_pred = reflection_pred + epsilon
                reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item

                restoration_imgs_pred = torch.zeros(*sun_imgs_x_qry[i].shape)
                restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_qry[:,0,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_qry[:,1,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_qry[:,2,:,:], reflection_pred[:,0,:,:])

                contour_loss = Contour_loss(K=5)
                weight_map = contour_loss.forward(sun_imgs_y_qry[i]).cpu()

                restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                gt_imgs = torch.mul(sun_imgs_y_qry[i], weight_map)

                mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
                
                ms_ssim_loss_fc = pytorch_msssim.MSSSIM(window_size = 7)
                ms_ssim_loss = ms_ssim_loss_fc(restoration_imgs_pred, gt_imgs)

                loss_q[k+1] = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ms_ssim_loss)) #+ REGULARIZATION * reg_loss
                
                # writer image
                self.writer.add_images('masks/true', gt_imgs, step)
                self.writer.add_images('masks/pred', restoration_imgs_pred, step)
 
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        
        # optimize
        self.meta_optim.step()

        return loss_q

    def finetunning(self, sun_imgs_x_spt, sun_imgs_y_spt, sun_imgs_x_qry, sun_imgs_y_qry, step):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            reflection_pred = net(sun_imgs_x_spt[i])
            reflection_pred = reflection_pred + epsilon
            reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
            
            restoration_imgs_pred = torch.zeros(*sun_imgs_x_spt.shape)
            restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_spt[:,0,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_spt[:,1,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_spt[:,2,:,:], reflection_pred[:,0,:,:])
            
            contour_loss = Contour_loss(K=5)
            weight_map = contour_loss.forward(sun_imgs_y_spt[i]).cpu()
            
            restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
            gt_imgs = torch.mul(sun_imgs_y_spt[i], weight_map)
            
            mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)

            grad = torch.autograd.grad(mse_loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                
                # 1. run the i-th task and compute loss for k=0
                reflection_pred = net(sun_imgs_x_qry[i], vars=None)
                reflection_pred = reflection_pred + epsilon
                reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item

                restoration_imgs_pred = torch.zeros(*sun_imgs_x_spt.shape)
                restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_spt[:,0,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_spt[:,1,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_spt[:,2,:,:], reflection_pred[:,0,:,:])

                contour_loss = Contour_loss(K=5)
                weight_map = contour_loss.forward(sun_imgs_y_spt[i]).cpu()

                restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                gt_imgs = torch.mul(sun_imgs_y_spt[i], weight_map)

                mse_loss_q[0] = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
                
                # writer image
                self.writer.add_images('masks/true', gt_imgs, step)
                self.writer.add_images('masks/pred', restoration_imgs_pred, step)
                
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                
                # 1. run the i-th task and compute loss for k=0
                reflection_pred = net(sun_imgs_x_qry[i], fast_weights)
                reflection_pred = reflection_pred + epsilon
                reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item

                restoration_imgs_pred = torch.zeros(*sun_imgs_x_qry.shape)
                restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_qry[:,0,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_qry[:,1,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_qry[:,2,:,:], reflection_pred[:,0,:,:])

                contour_loss = Contour_loss(K=5)
                weight_map = contour_loss.forward(sun_imgs_y_qry[i]).cpu()

                restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                gt_imgs = torch.mul(sun_imgs_y_qry[i], weight_map)

                mse_loss_q[1] = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
                
                # writer image
                self.writer.add_images('masks/true', gt_imgs, step)
                self.writer.add_images('masks/pred', restoration_imgs_pred, step)
                
            for k in range(1, self.update_step):
                
                # 1. run the i-th task and compute loss for k=0
                reflection_pred = self.net(sun_imgs_x_spt[i])
                reflection_pred = reflection_pred + epsilon
                reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item

                restoration_imgs_pred = torch.zeros(*sun_imgs_x_spt.shape)
                restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_spt[:,0,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_spt[:,1,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_spt[:,2,:,:], reflection_pred[:,0,:,:])

                contour_loss = Contour_loss(K=5)
                weight_map = contour_loss.forward(sun_imgs_y_spt[i]).cpu()

                restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                gt_imgs = torch.mul(sun_imgs_y_spt[i], weight_map)
                
                mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                
                reflection_pred = self.net(sun_imgs_x_qry[i], fast_weights)
                reflection_pred = reflection_pred + epsilon
                reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item

                restoration_imgs_pred = torch.zeros(*sun_imgs_x_qry.shape)
                restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_qry[:,0,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_qry[:,1,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_qry[:,2,:,:], reflection_pred[:,0,:,:])

                contour_loss = Contour_loss(K=5)
                weight_map = contour_loss.forward(sun_imgs_y_qry[i]).cpu()

                restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                gt_imgs = torch.mul(sun_imgs_y_qry[i], weight_map)

                mse_loss_q[k+1] = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
                
                # writer image
                self.writer.add_images('masks/true', gt_imgs, step)
                self.writer.add_images('masks/pred', restoration_imgs_pred, step)

        del net

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        return loss_q

def main():
    pass


if __name__ == '__main__':
    main()
