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

def train(maml, synthesis_train, synthesis_val, device, best_model_loss, writer, scheduler) :
    total_step = 0
    train_loss_all = 0
    for epoch in range(args.epoch):
        
        # fetch meta_batchsz num of episode each time
        train_loader = DataLoader(synthesis_train , args.task_num, shuffle=True, num_workers=4, pin_memory=True)
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_loader):
            total_step += 1
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            current_loss = maml(x_spt, y_spt, x_qry, y_qry, step)
            train_loss_all += current_loss.item()
            train_loss_all = train_loss_all/total_step
            if step % 10 == 0:
                print('INFO: [',epoch+1,'] -> Outer step:', step+1, 'average training loss:', train_loss_all)
                writer.add_scalar('Loss/Training_Step', train_loss_all, total_step)
            if step % 200 == 0 and step != 0:  # evaluation
                print("======================================")
                print('INFO: [',epoch+1,'] -> Finetunning...')
                val_loader = DataLoader(synthesis_val, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []
                finetune_loss_all = 0
                finetune_idx = 0
                for val_step, (x_spt, y_spt, x_qry, y_qry) in enumerate(val_loader):
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    current_loss = maml.finetunning(x_spt, y_spt, x_qry, y_qry, step)
                    finetune_loss_all += current_loss.item()
                    finetune_idx += 1
                finetune_loss_all =  finetune_loss_all/finetune_idx
                print('INFO: [',epoch+1,'] -> Outer step:', total_step, 'Finetunning average loss:', finetune_loss_all)
                if best_model_loss > finetune_loss_all :
                    best_model_loss = finetune_loss_all
                    torch.save(maml.net.state_dict(), args.dir_checkpoint + 'best_epoch.pth')
                    print("INFO: Saving Model at finetunning average loss : ", finetune_loss_all)
                print("============================================")
                writer.add_scalar('Loss/Finetune_Step', finetune_loss_all, total_step)

        if args.save :
            torch.save(maml.net.state_dict(), args.dir_checkpoint + 'final_epoch.pth')
        
        scheduler.step()
        writer.add_scalar('Loss/Epoch Training', train_loss_all, epoch+1)
        print("INFO: Finish {} epoch.\n".format(epoch+1))

def main():
    
    # set logging and writer
    writer = SummaryWriter(log_dir=args.dir_checkpoint+"run",comment=f'Outer Learning Rate_{args.meta_lr}Inner Learning Rate_{args.update_lr}_Task Number_{args.task_num}_Image Scale_{args.imgsz}')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    maml = Meta(args, writer, device).to(device)
    
    # define step scheduler
    scheduler = optim.lr_scheduler.StepLR(maml.meta_optim, step_size=args.step_size, gamma=args.step_adjust)
    
    # define global loss
    best_model_loss = 10000000

    # batch(batch set) of meta training set for each tasks and for meta testing
    synthesis_train = Synthesis_Image(args.dataset, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=15000, resize=args.imgsz, mat=args.mat)
    synthesis_val = Synthesis_Image(args.dataset, mode='validation', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=100, resize=args.imgsz, mat=args.mat)
    try:
        train(maml, synthesis_train, synthesis_val, device, best_model_loss, writer, scheduler)
    except KeyboardInterrupt:
        torch.save(maml.net.state_dict(), args.dir_checkpoint + 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=4)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=64)
    argparser.add_argument('--step_size', type=int, help='step size for adjusting learning rate', default=4)
    argparser.add_argument('--step_adjust', type=int, help='adjusting rate for adjusting learning rate', default=0.1)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--output_channel', type=int, help='output_channel', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=3)
    argparser.add_argument('--weight_decay', type=float, help='weight decay value of rmsprop optimizer', default=1e-8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
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