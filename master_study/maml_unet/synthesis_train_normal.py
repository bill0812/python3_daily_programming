import  torch, os
import  numpy as np
import  scipy.stats
import  random, sys, pickle
import  argparse
import logging

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from model.meta import Meta
from dataloader.synthesis_image import Synthesis_Image
from model.unet_model import UNet
from model.unet_model_relu_ms import UNet_Relu_MS
from utils.contour_loss import Contour_loss
from utils import pytorch_msssim

def train(net, synthesis_train, synthesis_val, device, best_model_loss, writer, scheduler, mse_loss_fn, ssim_loss_fc, contour_loss, meta_optim) :
    total_step = 0
    train_loss_all = 0
    # set epsilon
    epsilon=1e-10
    alpha = 0.12
    for epoch in range(args.epoch):
        

        # fetch meta_batchsz num of episode each time
        train_loader = DataLoader(synthesis_train , args.batchsize, shuffle=True, num_workers=4, pin_memory=True)
        net.train()
        for step, (sun_imgs, gt_imgs, name_input, name_truth) in enumerate(train_loader):
            total_step += 1
            sun_imgs, gt_imgs = sun_imgs.to(device), gt_imgs.to(device)

            reflection_pred = net(sun_imgs)
            reflection_pred = reflection_pred + epsilon
            reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
            restoration_imgs_pred = torch.zeros(*sun_imgs[:,:,:,:].shape).to(device)
            restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs[:,0,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs[:,1,:,:], reflection_pred[:,0,:,:])
            restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs[:,2,:,:], reflection_pred[:,0,:,:])
            
            weight_map = contour_loss.forward(gt_imgs)
            restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
            gt_imgs = torch.mul(gt_imgs, weight_map)
            
            mse_loss = mse_loss_fn(restoration_imgs_pred, gt_imgs)
            ssim_loss = ssim_loss_fc(restoration_imgs_pred, gt_imgs)
            current_loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss
            
            # optimize theta parameters
            meta_optim.zero_grad()
            current_loss.backward()
            meta_optim.step()
            torch.cuda.empty_cache()
            scheduler.step()
            
            # compute all loss
            train_loss_all += current_loss.item()
            train_loss_all = train_loss_all/(args.batchsize*total_step)

            if step % 10 == 0:
                print('INFO: [',epoch+1,'] -> Training Step:', step+1, 'average training loss:', train_loss_all)
                writer.add_scalar('Loss/Training_Step', train_loss_all, total_step)
            if step % 200 == 0 and step != 0:  # evaluation
                print("======================================")
                print('INFO: [',epoch+1,'] -> Validation...')
                val_loader = DataLoader(synthesis_val, args.batchsize, shuffle=True, num_workers=4, pin_memory=True)
                accs_all_test = []
                val_loss_all = 0
                val_idx = 0
                net.eval()
                for val_step, (sun_imgs, gt_imgs, name_input, name_truth) in enumerate(val_loader):
                    sun_imgs, gt_imgs = sun_imgs.to(device), gt_imgs.to(device)

                    with torch.no_grad() :
                        reflection_pred = net(sun_imgs)
                        reflection_pred = reflection_pred + epsilon
                        reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # Peter: may be we can try to remove this item
                        restoration_imgs_pred = torch.zeros(*sun_imgs[:,:,:,:].shape).to(device)
                        restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs[:,0,:,:], reflection_pred[:,0,:,:])
                        restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs[:,1,:,:], reflection_pred[:,0,:,:])
                        restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs[:,2,:,:], reflection_pred[:,0,:,:])
                        
                        weight_map = contour_loss.forward(gt_imgs)
                        restoration_imgs_pred = torch.mul(restoration_imgs_pred, weight_map)
                        gt_imgs = torch.mul(gt_imgs, weight_map)
                        
                        mse_loss = mse_loss_fn(restoration_imgs_pred, gt_imgs)
                        ssim_loss = ssim_loss_fc(restoration_imgs_pred, gt_imgs)
                        current_loss = torch.mean(mse_loss) + alpha* ( 1 - torch.mean(ssim_loss)) #+ REGULARIZATION * reg_loss
                        train_loss_all += current_loss.item()
                        train_loss_all = train_loss_all/total_step
                    
                    val_loss_all += current_loss.item()
                    val_idx += 1
                val_loss_all =  val_loss_all/val_idx
                print('INFO: [',epoch+1,'] -> Training Epoch:', total_step, 'Finetunning average loss:', val_loss_all)
                if best_model_loss > val_loss_all :
                    best_model_loss = val_loss_all
                    torch.save(net.state_dict(), args.dir_checkpoint + 'best_epoch.pth')
                    print("INFO: Saving Model at finetunning average loss : ", val_loss_all)
                print("============================================")
                writer.add_scalar('Loss/Finetune_Step', val_loss_all, total_step)
                torch.cuda.empty_cache()

        if args.save :
            torch.save(net.state_dict(), args.dir_checkpoint + 'final_epoch.pth')
        
        
        writer.add_scalar('Loss/Epoch Training', train_loss_all, epoch+1)
        print("INFO: Finish {} epoch.\n".format(epoch+1))

def main():
    
    # set logging and writer
    writer = SummaryWriter(log_dir=args.dir_checkpoint+"run",comment=f'Learning Rate_{args.meta_lr}_Batch size_{args.batchsize}_Image Scale_{args.imgsz}')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # define net
    net = UNet(in_channels=args.imgc, out_channels=args.output_channel)
    net.to(device)
    
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load)) 

    # define loss
    mse_loss_fn = torch.nn.MSELoss(reduction='none')
    ssim_loss_fc = pytorch_msssim.SSIM(window_size = 7)
    contour_loss = Contour_loss(K=5)
        
    # define optimizer for meta learning
    if args.optimizer == "adam" :
        meta_optim = optim.Adam(net.parameters(), lr=args.meta_lr)
    elif args.optimizer == "rmsprop":
        meta_optim = optim.RMSprop(net.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
    else :
        raise ValueError("Wrong Optimzer !")
    
    # define step scheduler
    scheduler = optim.lr_scheduler.StepLR(meta_optim, step_size=args.step_size, gamma=args.step_adjust)
    
    # define global loss
    best_model_loss = 10000000

    # batch(batch set) of meta training set for each tasks and for meta testing
    synthesis_train = Synthesis_Image(args.dataset, mode='train_normal', batchsz = args.batchsize)
    synthesis_val = Synthesis_Image(args.dataset, mode='val_normal', batchsz = args.batchsize)
    try:
        train(net, synthesis_train, synthesis_val, device, best_model_loss, writer, scheduler, mse_loss_fn, ssim_loss_fc, contour_loss, meta_optim)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), args.dir_checkpoint + 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=12)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=64)
    argparser.add_argument('--step_size', type=int, help='step size for adjusting learning rate', default=4)
    argparser.add_argument('--step_adjust', type=int, help='adjusting rate for adjusting learning rate', default=0.1)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--output_channel', type=int, help='output_channel', default=1)
    argparser.add_argument('--batchsize', type=int, help='batch size', default=4)
    argparser.add_argument('--weight_decay', type=float, help='weight decay value of rmsprop optimizer', default=1e-8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--optimizer', type=str, help='optimizer either adam or rmsprop', default="adam")
    argparser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')
    argparser.add_argument('--dir_checkpoint', type=str, help='path for saved model', default="model_saved/")
    argparser.add_argument('--dataset', type=str, help='path for dataset', default="dataset/maml_dataset/")
    argparser.add_argument('--mat', help='是否用 mat 檔的 mask 結合 Groundtruths 生成 Sun images？（用 mat 檔沒有 clip 問題！）', action='store_true')
    argparser.add_argument('--save', help='save epoch each epoch (overwrite)', action='store_true')
    args = argparser.parse_args()
    
    if not os.path.exists(args.dir_checkpoint):
        os.makedirs(args.dir_checkpoint)
        
    if not os.path.exists(args.dir_checkpoint+"run"):
        os.makedirs(args.dir_checkpoint+"run")
    
    # print all args
    print("INFO: All args are : \n================\n",args,"\n================ \nINFO: Below are some INFO and LOGS : \n================")
    
    # main function
    main()