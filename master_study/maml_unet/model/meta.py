# import basic module
import logging, numpy as np, sys

# import deeplearning module
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import copy

from model.unet_model import UNet
from model.unet_model_relu_ms import UNet_Relu_MS
from utils.contour_loss import Contour_loss
from utils import pytorch_msssim

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, writer, device):
        """

        :param args:
        """
        super(Meta, self).__init__()
        self.device = device
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.writer = writer
        self.in_channels = args.imgc
        self.out_channels = args.output_channel
        self.epsilon = 1e-10

        self.net = UNet(in_channels=args.imgc, out_channels=args.output_channel)
        self.net_param = []
        for param in self.net.parameters():
            self.net_param.append(param.clone().data.to(device))

        logging.info(f'Network:\n'
                 f'\t{self.net.in_channels} input channels\n'
                 f'\t{self.net.out_channels} output channels (classes)')
        
        if args.load:
            net.load_state_dict(
                torch.load(args.load, map_location=device)
            )
            logging.info(f'Model loaded from {args.load}')
        
        # define loss
        self.mse_loss_fn = torch.nn.MSELoss(reduction='none')
        self.ssim_loss_fc = pytorch_msssim.SSIM(window_size = 7)
        self.contour_loss = Contour_loss(K=5)
            
        # define optimizer for meta learning
        if args.optimizer == "adam" :
            self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        elif args.optimizer == "rmsprop":
            self.meta_optim = optim.RMSprop(self.net.parameters(), lr=self.meta_lr, weight_decay=self.weight_decay)
        else :
            raise ValueError("Wrong Optimzer !")

    def forward(self, sun_imgs_x_spt, sun_imgs_y_spt, sun_imgs_x_qry, sun_imgs_y_qry, step):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = sun_imgs_x_spt.size()
        querysz = sun_imgs_x_qry.size(1)

        loss_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        
        # set epsilon
        epsilon=1e-10
        alpha = 0.12
        self.net = self.net.to(self.device)
        self.net.train()
        for i in range(task_num):

            reflection_pred = self.net(sun_imgs_x_spt[i])
            reflection_pred = reflection_pred + epsilon
            reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
            restoration_imgs_pred = torch.zeros(*sun_imgs_x_spt[i][:,:,:,:].shape).to(self.device)
            restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_spt[i][:,0,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_spt[i][:,1,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_spt[i][:,2,:,:], reflection_pred[:,0,:,:])
            
            weight_map = self.contour_loss.forward(sun_imgs_y_spt[i])
            restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
            gt_imgs = torch.mul(sun_imgs_y_spt[i], weight_map)
            
            mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
            ssim_loss = self.ssim_loss_fc(restoration_imgs_pred, gt_imgs)
            loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss
            
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # =====================================================

            reflection_pred = self.net(sun_imgs_x_qry[i])
            reflection_pred = reflection_pred + epsilon
            reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
            restoration_imgs_pred = torch.zeros(*sun_imgs_x_qry[i][:,:,:,:].shape).to(self.device)
            restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_qry[i][:,0,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_qry[i][:,1,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_qry[i][:,2,:,:], reflection_pred[:,0,:,:])
            
            weight_map = self.contour_loss.forward(sun_imgs_y_qry[i])
            restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
            gt_imgs = torch.mul(sun_imgs_y_qry[i], weight_map)

            mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
            ssim_loss = self.ssim_loss_fc(restoration_imgs_pred, gt_imgs)
            loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss
            loss_q[0] += loss

            for each_fastweight, (name, param) in zip(fast_weights, self.net.named_parameters()):
                param.data = each_fastweight.data

            # =====================================================

            reflection_pred = self.net(sun_imgs_x_qry[i])
            reflection_pred = reflection_pred + epsilon
            reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
            restoration_imgs_pred = torch.zeros(*sun_imgs_x_qry[i][:,:,:,:].shape).to(self.device)
            restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_qry[i][:,0,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_qry[i][:,1,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_qry[i][:,2,:,:], reflection_pred[:,0,:,:])
            
            weight_map = self.contour_loss.forward(sun_imgs_y_qry[i])
            restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
            gt_imgs = torch.mul(sun_imgs_y_qry[i], weight_map)

            mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
            ssim_loss = self.ssim_loss_fc(restoration_imgs_pred, gt_imgs)
            loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss
            loss_q[1] += loss

            del grad, fast_weights, loss, mse_loss, ssim_loss, weight_map

            # =====================================================

            for k in range(1, self.update_step):
                reflection_pred = self.net(sun_imgs_x_spt[i])
                reflection_pred = reflection_pred + epsilon
                reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
                restoration_imgs_pred = torch.zeros(*sun_imgs_x_spt[i][:,:,:,:].shape).to(self.device)
                restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_spt[i][:,0,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_spt[i][:,1,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_spt[i][:,2,:,:], reflection_pred[:,0,:,:])
                
                weight_map = self.contour_loss.forward(sun_imgs_y_spt[i])
                restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                gt_imgs = torch.mul(sun_imgs_y_spt[i], weight_map)

                mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
                ssim_loss = self.ssim_loss_fc(restoration_imgs_pred, gt_imgs)
                loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss
                
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

                for each_fastweight, (name, param) in zip(fast_weights, self.net.named_parameters()):
                    param.data = each_fastweight.data
                                
                reflection_pred = self.net(sun_imgs_x_qry[i])
                reflection_pred = reflection_pred + epsilon
                reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
                restoration_imgs_pred = torch.zeros(*sun_imgs_x_qry[i][:,:,:,:].shape).to(self.device)
                restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_qry[i][:,0,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_qry[i][:,1,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_qry[i][:,2,:,:], reflection_pred[:,0,:,:])
                
                weight_map = self.contour_loss.forward(sun_imgs_y_qry[i])
                restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                gt_imgs = torch.mul(sun_imgs_y_qry[i], weight_map)

                mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
                ssim_loss = self.ssim_loss_fc(restoration_imgs_pred, gt_imgs)
                loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss
                loss_q[k+1] += loss

                del grad, fast_weights, loss, mse_loss, ssim_loss, weight_map
        
        for net_param, (name, param) in zip(self.net_param, self.net.named_parameters()):
            param.data = net_param
        
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = torch.sum(torch.stack(loss_q)) / task_num      

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        
        # optimize
        self.meta_optim.step()
        
        self.net_param = []
        for param in self.net.parameters():
            self.net_param.append(param.clone().data)

        torch.cuda.empty_cache()

        return loss_q

    def finetunning(self, sun_imgs_x_spt, sun_imgs_y_spt, sun_imgs_x_qry, sun_imgs_y_qry, step):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [setsz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        setsz, c_, h, w = sun_imgs_x_spt.size()
        querysz = sun_imgs_x_qry.size(1)

        loss_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        
        # set epsilon
        epsilon=1e-10
        alpha = 0.12
        self.net = self.net.to(self.device)
        self.net.train()
            
        # 1. run the i-th task and compute loss for k=0
        reflection_pred = self.net(sun_imgs_x_spt)
        reflection_pred = reflection_pred + epsilon
        reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
        restoration_imgs_pred = torch.zeros(*sun_imgs_x_spt[:,:,:,:].shape).to(self.device)
        restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_spt[:,0,:,:], reflection_pred[:,0,:,:])
        restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_spt[:,1,:,:], reflection_pred[:,0,:,:])
        restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_spt[:,2,:,:], reflection_pred[:,0,:,:])

        weight_map = self.contour_loss.forward(sun_imgs_y_spt)
        restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
        gt_imgs = torch.mul(sun_imgs_y_spt, weight_map)

        mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
        ssim_loss = self.ssim_loss_fc(restoration_imgs_pred, gt_imgs)
        loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss

        grad = torch.autograd.grad(loss, self.net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        with torch.no_grad():
                
            # 1. run the i-th task and compute loss for k=0
            reflection_pred = self.net(sun_imgs_x_qry)
            reflection_pred = reflection_pred + epsilon
            reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
            restoration_imgs_pred = torch.zeros(*sun_imgs_x_qry[:,:,:,:].shape).to(self.device)
            restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_qry[:,0,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_qry[:,1,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_qry[:,2,:,:], reflection_pred[:,0,:,:])

            weight_map = self.contour_loss.forward(sun_imgs_y_qry)
            restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
            gt_imgs = torch.mul(sun_imgs_y_qry, weight_map)

            mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
            ssim_loss = self.ssim_loss_fc(restoration_imgs_pred, gt_imgs)
            loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss
            loss_q[0] += loss

        for each_fastweight, (name, param) in zip(fast_weights, self.net.named_parameters()):
            param = each_fastweight

        with torch.no_grad():

            # 1. run the i-th task and compute loss for k=0
            # not use original net, use copy one
            reflection_pred = self.net(sun_imgs_x_qry)
            reflection_pred = reflection_pred + epsilon
            reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
            restoration_imgs_pred = torch.zeros(*sun_imgs_x_qry[:,:,:,:].shape).to(self.device)
            restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_qry[:,0,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_qry[:,1,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_qry[:,2,:,:], reflection_pred[:,0,:,:])

            weight_map = self.contour_loss.forward(sun_imgs_y_qry)
            restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
            gt_imgs = torch.mul(sun_imgs_y_qry, weight_map)

            mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
            ssim_loss = self.ssim_loss_fc(restoration_imgs_pred, gt_imgs)
            loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss
            loss_q[1] += loss

        del grad, fast_weights, loss, mse_loss, ssim_loss, weight_map

        for k in range(1, self.update_step_test):

            reflection_pred = self.net(sun_imgs_x_spt)
            reflection_pred = reflection_pred + epsilon
            reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
            restoration_imgs_pred = torch.zeros(*sun_imgs_x_spt[:,:,:,:].shape).to(self.device)
            restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_spt[:,0,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_spt[:,1,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_spt[:,2,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred = torch.clamp(reflection_pred, 0, 1)

            weight_map = self.contour_loss.forward(sun_imgs_y_spt)
            restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
            gt_imgs = torch.mul(sun_imgs_y_spt, weight_map)

            mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
            ssim_loss = self.ssim_loss_fc(restoration_imgs_pred, gt_imgs)
            loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss

            grad = torch.autograd.grad(loss, self.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            for each_fastweight, (name, param) in zip(fast_weights, self.net.named_parameters()):
                param = each_fastweight

            with torch.no_grad():
                reflection_pred = self.net(sun_imgs_x_qry)
                reflection_pred = reflection_pred + epsilon
                reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
                restoration_imgs_pred = torch.zeros(*sun_imgs_x_qry[:,:,:,:].shape).to(self.device)
                restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs_x_qry[:,0,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs_x_qry[:,1,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs_x_qry[:,2,:,:], reflection_pred[:,0,:,:])
                restoration_imgs_pred = torch.clamp(reflection_pred, 0, 1)

                weight_map = self.contour_loss.forward(sun_imgs_y_qry)
                restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                gt_imgs = torch.mul(sun_imgs_y_qry, weight_map)

                mse_loss = self.mse_loss_fn(restoration_imgs_pred, gt_imgs)
                ssim_loss = self.ssim_loss_fc(restoration_imgs_pred, gt_imgs)
                loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss
                loss_q[k+1] += loss

            del grad, fast_weights, loss, mse_loss, ssim_loss, weight_map
        
        # this is the loss and accuracy before first update
        for net_param, (name, param) in zip(self.net_param, self.net.named_parameters()):
            param = net_param
        
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = torch.sum(torch.stack(loss_q))

        torch.cuda.empty_cache()
        
        return loss_q

def main():
    pass

if __name__ == '__main__':
    main()
