import time, os, sys, torch, visdom, argparse, math
import numpy as np
from pprint import PrettyPrinter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

# import mine/reference packages
from SSD_utils.data import *
from SSD_utils.model_Bayesian_Softplus import build_ssd, MultiBoxLoss
from SSD_utils.data.augmentations import SSDAugmentation
from SSD_utils.data.utils import *

# ===========================================
# from 21 epoch, i start adjust learning rate in 600,1200,1800 iter
# but before, i didn't adjust that, just stay in a certain rate of default
# current detect result is pretty bad in validation and test
# but not bad in training
# ===========================================

DATASET_ROOT = "/media/bill/BillData/比爾Bayesian-Satillite /dataset/Xview/"

# set up some details for the training and testing model
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

parser = argparse.ArgumentParser(description=\
    'Single Shot MultiBox Detector Training for XVIEW Dataset With Pytorch')

# set model others parameters
parser.add_argument('--batch_size', default=8, type=int,
					help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
					help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, action='store_false',
					help='Do not use CUDA to train model')
parser.add_argument('--num_epochs', default=50, type=int,
					help='Number of epochs on the data')
parser.add_argument('--num_samples', default=1, type=int, help='Number of samples')
parser.add_argument('--print_freq', default=10, type=int,
					help='Print training or validation status every __ batches')
parser.add_argument('--start_global_step', default=0, type=int,
					help='Resume global_step value at this step.')
parser.add_argument('--validate', default=False, action='store_true',
					help='If set to True, validation scores are calculated.')
parser.add_argument('--start_iter', default=0, type=int,
					help='Resume training at this iter')
parser.add_argument('--checkpoint', default=None, type=str,
					help='Checkpoint state_dict file to resume training from')
parser.add_argument('--saved_pretrained', default='pretrained_vgg/',
					help='Pretrained base model filename.')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
					help='Pretrained base model filename.')

# set variable for optimizers
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
					help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float,
					help='Weight decay for Adam')
parser.add_argument('--gamma', default=0.1, type=float,
					help='Gamma update for Adam')
parser.add_argument('--grad_clip', action='store_true',
					help='Using gradient clip for Gradient Descent')
parser.add_argument('--beta_type', default="Blundell", type=str, help='Beta type')

# data using in model include training/validation data and ouput directory
parser.add_argument('--dataset', default='XVIEW', choices=['XVIEW'],
					type=str, help='Name of Dataset. Choiceseee - [XVIEW]')
parser.add_argument('--dataset_root', default=XVIEW_ROOT,
					help='Dataset root directory path')
parser.add_argument('--training_data', type=str,
					default='dataset_400_new/training_data.csv',
					help='location of training_data')
parser.add_argument('--validation_data', type=str,
                    default='dataset_400_new/validation_data.csv',
					help='location of validation_data')
parser.add_argument('--save_folder', default='../outputs/',
					help='Directory for saving checkpoint models')
parser.add_argument('--print_to_file', default=False, action='store_true',
					help='If set to True, outputs are printed to file. \
						  Requires the print_filename argument to be given.')
parser.add_argument('--print_filename', default="../outputs/log.txt", type=str,
					help='File path and name to store the outputs.')

args = parser.parse_args()
# ========================================================================

# set up some environment and variables, also making directory
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
	if args.cuda:
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
	if not args.cuda:
		print("WARNING: It looks like you have a CUDA device, but aren't " +
			  "using CUDA.\nRun with --cuda for optimal training speed.")

if not os.path.exists(args.save_folder):
	os.mkdir(args.save_folder)

if args.print_to_file:
	f = open(args.print_filename, 'w')

start_epoch = 0
global_step_train = 0
global_step_val = 0
loss = 10000
loss_c = 10000
loss_l = 10000
step_index = 0
# ===============================================

# main to iter each epoch for training and validation
def main():
    """
    Training and validation.
    """

    global start_epoch
    global cfg
    global loss, loss_c, loss_l
    global best_loss

    # set up for cuda device
    if torch.cuda.is_available():
        if args.cuda:
            device = torch.device("cuda")      
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                "using CUDA.\nRun with --cuda for optimal training speed.")

    # Custom dataloaders
    if args.dataset == 'XVIEW':
        if args.dataset_root != XVIEW_ROOT:
            if not os.path.exists(XVIEW_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            
        cfg = xview

        print("| Preparing Dataset ...")

        # rename the file name
        args.training_data = DATASET_ROOT + args.training_data
        args.validation_data = DATASET_ROOT + args.validation_data

        # load training dataset and train data loader
        train_dataset = XVIEWDetection(args.training_data,\
            transform=SSDAugmentation(cfg['min_dim'],MEANS)) # we will use SSD augmentation in the future, using SSDAugmentation

        train_data_loader = DataLoader(train_dataset, args.batch_size,\
            num_workers=args.num_workers,\
            shuffle=True, collate_fn=detection_collate,\
            pin_memory=True)

        # ===================================================

        # load validation dataset and validation data loader
        val_dataset = XVIEWDetection(args.validation_data,\
            transform=SSDAugmentation(cfg['min_dim'],MEANS)) # we will use SSD augmentation in the future, using SSDAugmentation
        
        val_data_loader = DataLoader(val_dataset, args.batch_size,\
            num_workers=args.num_workers,\
            shuffle=True, collate_fn=detection_collate,\
            pin_memory=True)
        # ===================================================

    # Initialize model or load checkpoint
    print(cfg)
    model = build_ssd(cfg['min_dim'], cfg['num_classes'])
    if args.checkpoint is None:
        
        # loading base vgg model
        # vgg_weights = torch.load(args.saved_pretrained + args.basenet)
        # print('\n| Loading base network...')
        # model.vgg.load_state_dict(vgg_weights)
        
        # initialize newly added layers' weights with xavier method
        # print('\n| Initializing weights...')
        # model.extras.apply(weights_init)
        # model.loc.apply(weights_init)
        # model.conf.apply(weights_init)

        # declare optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # define best loss
        best_loss = 10000
    else:
        print('\n| Resuming training, loading {}...'.format(args.checkpoint))
        check_point_dict = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        model.load_weights(args.checkpoint, check_point_dict)

        # declare optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(check_point_dict['optimizer_state_dict'])
        best_loss = check_point_dict["best_loss"]
        start_epoch = check_point_dict["epoch"]

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss().to(device)
    # criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    
    for epoch in range(start_epoch, args.num_epochs):
        # Paper describes decaying the learning rate at the 80000th, 100000th, 120000th 'iteration', i.e. model update or batch
        # The paper uses a batch size of 32, which means there were about 517 iterations in an epoch
        # Therefore, to find the epochs to decay at, you could do,

        # print("\n| Training ==>")

        # # One epoch's training
        # loss, loss_c, loss_l, loss_kl = train(train_loader=train_data_loader,
        #                     model=model,
        #                     criterion=criterion,
        #                     optimizer=optimizer,
        #                     epoch=epoch, global_epoch = global_step_train,cfg=cfg)

        # # Did loss improve?
        # is_best = loss < best_loss
        # best_loss = min(loss, best_loss)

        # # save model every 3 epoch
        # # if epoch % 3 == 0 and epoch != 0:
        # print('! Saving state, epoch:', epoch+1)
        # state_dict_filename = args.dataset + '_SSD_'  + 'new.pt'
        # state_dict_filename_best = args.dataset + '_SSD_' + 'best.pt'
        # state = {
        #         'epoch': epoch,
        #         'kl divergence' : loss_kl,
        #         'loss': loss,
        #         'best_loss': best_loss,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict()
        #         }
        # torch.save(state,
        #     args.save_folder + state_dict_filename)

        # if is_best :
        #     torch.save(state,
        #         args.save_folder + state_dict_filename_best)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print("| Validation ==>")
            validate(val_loader=val_data_loader,
                        model=model,
                        criterion=criterion,
                        epoch=epoch,global_epoch = global_step_val)

# =======================================
def train(train_loader, model, criterion, optimizer, epoch, global_epoch, cfg):

    global global_step_train
    global step_index

    # set up model's train mode
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_l = AverageMeter()
    losses_c = AverageMeter()
    losses_kl = AverageMeter()
    losses = AverageMeter()

    start = time.time()
    m = math.ceil(len(train_loader)/args.batch_size)
    # Batches
    for i, (images, ground_truth) in enumerate(train_loader):

        # if i == 50 :
        #     break

        if args.beta_type is "Blundell":
            beta = 2 ** (m - (i + 1)) / (2 ** m - 1)
        elif args.beta_type is "Soenderby":
            beta = min(epoch / (num_epochs//4), 1)
        elif args.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0

        if i in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        global_step_train += 1

        data_time.update(time.time() - start)

        # Move to default device
        images = Variable(images.cuda())
        images = images.repeat(args.num_samples,1,1,1)
        ground_truth = [Variable(ann.cuda(), requires_grad=True) for ann in ground_truth]
        ground_truth = ground_truth * args.num_samples
        
        # Forward prop.
        predicted_locs, predicted_scores, prior, total_kl = model(images)

        # Loss  
        boxes = [i[:, :-1].cuda() for i in ground_truth]
        labels = [i[:,-1].cuda() for i in ground_truth]
        loss_l, loss_c = criterion(predicted_locs, predicted_scores, prior, boxes, labels)
        loss_lc = loss_l + loss_c
        total_kl = total_kl.mean()
        loss = loss_lc.mean() + total_kl.mean()

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if args.grad_clip :
            clip_gradient(optimizer, 10)
            # clip_grad_norm_(model.parameters(), 5, norm_type=1)

        # Update model
        optimizer.step()
        losses_c.update(loss_c.item(), images.size(0))
        losses_l.update(loss_l.item(), images.size(0))
        losses_kl.update(total_kl.item(), images.size(0))
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            print('| Epoch: [%d / %d][%d / %d]  /  '
                  'Iter Time %.3f /  '
                  'Data Time %.3f \n'
                  '| Confidence Loss %.4f /  '
                  'Coordinates Loss %.4f /  KL Divergence %.4f / '
                  'Total Loss %.4f '
                  '\n===================================================================\n' %(epoch+1, args.num_epochs, i, len(train_loader),
                                        batch_time.val, data_time.val\
                                        ,losses_c.val ,losses_l.val,losses.val, losses_kl.val), end="")

            if args.print_to_file:
                f.write('| Epoch: [{0} / {1}][{2} / {3}]  /  '
                  'Iter Time {batch_time.val:.3f} /  '
                  'Data Time {data_time.val:.3f} \n'
                  '| Confidence Loss {loss_c.val:.4f} /  '
                  'Coordinates Loss {loss_l.val:.4f} / KL Divergence {loss_kl.val:.4f} '
                  'Total Loss {loss.val:.4f} '
                  '\n==================================================================='.format(epoch+1, args.num_epochs, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss_c=losses_c, loss_l=losses_l, loss_kl=losses_kl, loss=losses))
        
        del loss_c, loss_l, loss, images, ground_truth, boxes, labels, total_kl, loss_lc
        del predicted_locs, predicted_scores, prior

        torch.cuda.empty_cache()

    # return loss to save best model
    return losses.val, losses_c.val, losses_l.val, losses_kl

def validate(val_loader, model, criterion, epoch, global_epoch):

    global global_step_val

    # set up eval
    model.eval()  # eval mode disables dropout

    batch_time = AverageMeter()
    losses_l = AverageMeter()
    losses_c = AverageMeter()
    losses_kl = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # set for caculating mAP
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    with torch.no_grad():
        m = math.ceil(len(val_loader)/args.batch_size)
        # Batches
        for i, (images, ground_truth) in enumerate(val_loader):

            # if i == 50 :
            #     break

            if args.beta_type is "Blundell":
                beta = 2 ** (m - (i + 1)) / (2 ** m - 1)
            elif args.beta_type is "Soenderby":
                beta = min(epoch / (num_epochs//4), 1)
            elif args.beta_type is "Standard":
                beta = 1 / m
            else:
                beta = 0

            global_step_val += 1

            # Move to default device
            images = Variable(images.cuda())
            images = images.repeat(args.num_samples,1,1,1)
            ground_truth = [Variable(ann.cuda(), requires_grad=True) for ann in ground_truth]
            ground_truth = ground_truth * args.num_samples
        
            # Forward prop.
            predicted_locs, predicted_scores, prior, total_kl = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
            # predicted_scores = F.softmax(predicted_scores, -1)

            # Loss
            boxes = [i[:, :-1].cuda() for i in ground_truth]
            labels = [i[:,-1].cuda() for i in ground_truth]
            loss_l, loss_c = criterion(predicted_locs, predicted_scores, prior, boxes, labels)  # scalar
            loss_lc = loss_l + loss_c
            total_kl = total_kl.mean()
            loss = loss_lc.mean() + total_kl.mean()

            
            losses_c.update(loss_c.item(), images.size(0))
            losses_l.update(loss_l.item(), images.size(0))
            losses_kl.update(total_kl.item(), images.size(0))
            losses.update(loss.item(), images.size(0))

            # # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores, prior,
                                                                                    min_score=0.05, max_overlap=0.5,
                                                                                    top_k=50)

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)

            del images, ground_truth, det_boxes_batch, det_labels_batch, det_scores_batch, boxes, labels
            del predicted_locs, predicted_scores, prior, total_kl

            batch_time.update(time.time() - start)
            start = time.time()
            # Print status
            if i % args.print_freq == 0:
                print('\r| [ %d / %d][%d / %d] / '
                    'Iter Time %.3f / Coordinate loss : %.3f / Classification loss : %.3f / KL Divergence : %.3f / Total Loss : %.3f' %(epoch+1, args.num_epochs,i, len(val_loader),
                                    batch_time.val, losses_c.val, losses_l.val, losses_kl.val, losses.val),end="")
                sys.stdout.flush()

            torch.cuda.empty_cache()

    # # Calculate mAP
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)

    # Print AP for each class
    print("==============================")
    print("APs : =>")
    pp.pprint(APs)
    print("\nMean Average Precision : => {:.3f}".format(mAP))
    print("==============================")

    del APs, mAP
    torch.cuda.empty_cache()

def adjust_learning_rate(optimizer, gamma, step):
	"""Sets the learning rate to the initial LR decayed by 10 at every
		specified step
	# Adapted from PyTorch Imagenet example:
	# https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	lr = args.lr * (gamma ** (step))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def xavier(param):
	init.xavier_uniform_(param)

def weights_init(m):
	if isinstance(m, nn.Conv2d):
		xavier(m.weight.data)
		m.bias.data.zero_()

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
# ==================================================================

if __name__ == '__main__':
    main()
