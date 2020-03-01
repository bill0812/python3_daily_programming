import  torch, os
import  numpy as np
import  scipy.stats
import  random, sys, pickle
import  argparse
import logging
import cv2
from cv2.ximgproc import guidedFilter

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from model.meta import Meta
from dataloader.synthesis_image import Synthesis_Image
from model.unet_model import UNet
from utils.contour_loss import Contour_loss
from utils import pytorch_msssim

def test(net, mse_loss_fn, ssim_loss_fc, contour_loss, synthesis_test, device, writer) :
    epsilon = 1e-10
    test_loader = DataLoader(synthesis_test , args.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    mse_score_all = 0
    ssim_score_all = 0
    for idx, (sun_imgs ,gt_imgs, name_input, name_truth) in enumerate(test_loader):
        
        if args.gpu:
            sun_imgs = sun_imgs.cuda()
            gt_imgs = gt_imgs.cuda()

        reflection_pred = net(sun_imgs) 
        reflection_pred = reflection_pred + epsilon # for all lum
        reflection_pred = torch.clamp(reflection_pred, 0.1, 5.0) # for all lum
        restoration_imgs_pred = torch.zeros(*sun_imgs[:,:,:,:].shape).to(device)
        restoration_imgs_pred[:,0,:,:] = torch.div(sun_imgs[:,0,:,:], reflection_pred[:,0,:,:])
        restoration_imgs_pred[:,1,:,:] = torch.div(sun_imgs[:,1,:,:], reflection_pred[:,0,:,:])
        restoration_imgs_pred[:,2,:,:] = torch.div(sun_imgs[:,2,:,:], reflection_pred[:,0,:,:])
        
        # compute score
        weight_map = contour_loss.forward(gt_imgs)
        restoration_imgs_pred_contour = torch.mul(restoration_imgs_pred, weight_map)
        gt_imgs_contour = torch.mul(gt_imgs, weight_map)
        mse_score = torch.mean(mse_loss_fn(restoration_imgs_pred_contour, gt_imgs_contour))
        ssim_score = ssim_loss_fc(restoration_imgs_pred_contour, gt_imgs_contour)
        ssim_score = torch.mean(ssim_score)
        mse_score_all += mse_score.item()
        ssim_score_all += ssim_score.item()
        print('Current MSE Average Score : ',mse_score_all/(idx+1))
        print('Current SSIM Average Score : ',ssim_score_all/(idx+1))
        writer.add_scalar('Loss/MSE_Average_Score', mse_score_all/(idx+1), idx+1)
        writer.add_scalar('Loss/SSIM_Average_Score', ssim_score_all/(idx+1), idx+1)
        
        reflection_pred = reflection_pred.permute(0,2,3,1).cpu().data.numpy()
        reflection_pred = np.squeeze(reflection_pred)
        reflection_pred = np.where(reflection_pred>0, reflection_pred, 0.0000000001)
        
        gt_imgs = gt_imgs[0]
        sun_imgs = sun_imgs[0]

        # save gt and sun images
        sun_imgs = np.squeeze(sun_imgs.cpu().data.numpy())
        sun_imgs = np.transpose(sun_imgs, (1,2,0))
        sun_imgs_save = sun_imgs*255
        sun_imgs_save = sun_imgs_save[...,::-1]
        cv2.imwrite(args.result + name_input[0],sun_imgs_save.astype(np.uint8))
        gt_imgs = np.squeeze(gt_imgs.cpu().data.numpy())
        gt_imgs = np.transpose(gt_imgs, (1,2,0))
        gt_imgs = gt_imgs*255
        gt_imgs = gt_imgs[...,::-1]
        cv2.imwrite(args.result + name_truth[0],gt_imgs.astype(np.uint8))
        
        # save predict result
        reflection_pred_guided = guidedFilter(guide=reflection_pred, src=sun_imgs, radius=30, eps=1e-8, dDepth=-1)
        reflection_pred_normal = cv2.cvtColor(reflection_pred, cv2.COLOR_GRAY2RGB)
        reflection_pred_guided = (reflection_pred_guided - np.min(reflection_pred_guided)) / ( np.max(reflection_pred_guided) - np.min(reflection_pred_guided))
        reflection_pred_guided = reflection_pred_guided *1.4 + 0.2
        reflection_pred_normal = (reflection_pred_normal - np.min(reflection_pred_normal)) / ( np.max(reflection_pred_normal) - np.min(reflection_pred_normal))
        reflection_pred_normal = reflection_pred_normal *1.4 + 0.2
        reflection_pred_guided = np.where(reflection_pred_guided == 0, 1, reflection_pred_guided)
        reflection_pred_normal = np.where(reflection_pred_normal == 0, 1, reflection_pred_normal)
        reflection_pred_guided = np.divide(sun_imgs, reflection_pred_guided)
        reflection_pred_normal = np.divide(sun_imgs, reflection_pred_normal)
        
        # strech
        reflection_pred_guided = (reflection_pred_guided - np.min(reflection_pred_guided)) / ( np.max(reflection_pred_guided) - np.min(reflection_pred_guided))
        reflection_pred_normal = (reflection_pred_normal - np.min(reflection_pred_normal)) / ( np.max(reflection_pred_normal) - np.min(reflection_pred_normal))
        reflection_pred_guided = reflection_pred_guided*255
        reflection_pred_normal = reflection_pred_normal*255
        reflection_pred_normal = reflection_pred_normal[...,::-1]
        reflection_pred_guided = reflection_pred_guided[...,::-1]
        reflection_pred_guided = np.clip(reflection_pred_guided, 0, 255)
        reflection_pred_normal = np.clip(reflection_pred_normal, 0, 255)

        cv2.imwrite(args.result + "guided_" +name_truth[0],reflection_pred_guided.astype(np.uint8))
        cv2.imwrite(args.result + "normal_" +name_truth[0],reflection_pred_normal.astype(np.uint8))
    

def main():
    
    # set logging and writer
    writer = SummaryWriter(log_dir=args.dir_checkpoint)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    synthesis_test = Synthesis_Image(args.dataset, mode='test', batchsz = args.batchsize, test_data=args.test_data)

    # define loss
    mse_loss_fn = torch.nn.MSELoss(reduction='none')
    ssim_loss_fc = pytorch_msssim.SSIM(window_size = 7)
    contour_loss = Contour_loss(K=5)

    # define net
    net = UNet(in_channels=args.imgc, out_channels=args.output_channel)
    net.to(device)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load)) 

    try:
        test(net,mse_loss_fn,ssim_loss_fc,contour_loss,synthesis_test,device,writer)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', action='store_true', help='use cuda')
    argparser.add_argument('--batchsize', type=int, help='batch size', default=1)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=64)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--output_channel', type=int, help='output_channel', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=3)
    argparser.add_argument('--load', type=str, default="run_history/maml_final_1/best_epoch.pth", help='Load model from a .pth file')
    argparser.add_argument('--dir_checkpoint', type=str, help='path for saved run', default="run_1/")
    argparser.add_argument('--result', type=str, help='path for saved result', default="predict_result_withMaml/test_test/")
    argparser.add_argument('--test_data', type=str, help='dataset for testing', default="test")
    argparser.add_argument('--dataset', type=str, help='path for dataset', default="dataset/maml_dataset/")
    args = argparser.parse_args()
    
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    # main function
    main()
