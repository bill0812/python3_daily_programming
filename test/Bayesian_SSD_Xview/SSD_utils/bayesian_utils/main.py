'''
reference from https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/main.py
and revise to Bayesian Version, thanks to this repo
'''
import argparse
import os
import shutil
import time
import math
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim

import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# import vgg_nonBayes as vgg
# from vgg_BayesBackProp import VGG as vgg_BB
import bayesian_config as cf
from Original.BBBlayers import GaussianVariationalInference
import vgg_nonBayes as vgg
import vgg_BayesSoftplus as vgg_SP
import vgg_BayesBackProp as vgg_BB

model_names = sorted(name for name in vgg_SP.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg_SP.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.00001, type=float, help='learning_rate')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
#parser.add_argument('--depth', default=28, type=int, help='depth of model')
#parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--num_samples', default=10, type=int, help='Number of samples')
parser.add_argument('--beta_type', default="Blundell", type=str, help='Beta type')
parser.add_argument('--p_logvar_init', default=0, type=int, help='p_logvar_init')
parser.add_argument('--q_logvar_init', default=-10, type=int, help='q_logvar_init')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [mnist/cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--save_folder', default='outputs_SP_paper/', help='Directory for saving checkpoint models')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()

if not os.path.exists(args.save_folder):
	os.mkdir(args.save_folder)

if use_cuda is True:
    torch.cuda.set_device(0)
best_acc = 0
resize = 32
valid_size = 0.2
start_epoch, num_epochs, batch_size, optim_type, classes = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type, cf.classes
valid_loss_min = np.Inf # track change in validation loss

# Data Uplaod
print('\n[Phase 1] : Data Preparation')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

valid_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True)

# Return network & file name
def getNetwork(args):
    model = vgg_SP.__dict__[args.arch]()
    # model = vgg.__dict__[args.arch]()
    # model = vgg_BB.__dict__[args.arch]()
    print("| Model : ",model,"\n=======================================")

    return model, args.arch + "-"


# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir(args.save_folder + '/checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type ' + args.arch + '...')
    net, file_name = getNetwork(args)

if use_cuda:
    net.cuda()

vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())
# vi = nn.CrossEntropyLoss()
logfile = os.path.join(args.save_folder + '/diagnostics_Bayes{}_{}.txt'.format(args.arch, args.dataset))
val_logfile = os.path.join(args.save_folder + "/val_diagnostics_Bayes{}_{}.txt".format(args.arch, args.dataset))
value_file = os.path.join(args.save_folder + "/values{}_{}.txt".format(args.arch, args.dataset))

def train_and_val(epoch):

    ###################
    # train the model #
    ####################
    likelihoods = []
    kls = []
    net.train()
    avg_train_loss = 0
    train_loss = 0
    valid_loss = 0
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0
    accuracy_train = 0
    global valid_loss_min
    m = math.ceil(len(train_loader) / batch_size)
    optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate(args.lr, epoch), weight_decay=args.weight_decay)
    print('\n| Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.view(-1, 3, resize, resize).repeat(args.num_samples, 1, 1, 1)
        y = y.repeat(args.num_samples)

        if use_cuda:
            x, y = x.cuda(), y.cuda() # GPU settings

        if args.beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif args.beta_type is "Soenderby":
            beta = min(epoch / (num_epochs // 4), 1)
        elif args.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0

        # Forward Propagation
        x, y = Variable(x), Variable(y)
        # outputs, loss_train = net(x, y, args.num_samples, batch_size, 10, "train")
        outputs, kl = net(x)
        outputs = normalization_function(outputs)
        loss_train = vi(outputs, y, kl, beta)
        ll = loss_train.data.mean() - beta*kl.data
        
        train_loss += loss_train.item()

        optimizer.zero_grad()
        loss_train.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update
        
        _, predicted = outputs.max(1)
        accuracy_train = (predicted.data.cpu() == y.cpu()).float().mean()
        total_train += y.size(0)

        kls.append(beta*kl)
        likelihoods.append(ll)
        avg_train_loss = train_loss/total_train

        # print training/validation statistics 
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] Average Training Loss: %.4f Average Training Accuracy: %.3f Average KL : %.4f Average Likelihood : %.4f' %(epoch, num_epochs, batch_idx+1,
                    len(train_loader), avg_train_loss , accuracy_train, sum(kls)/len(kls), sum(likelihoods)/len(likelihoods)))
        sys.stdout.flush()

    ######################    
    # validate the model #
    ######################
    conf = []
    likelihoods_val = []
    kls_val = []
    average_loss = 0
    accuracy_val = 0
    net.eval()
    m = math.ceil(len(valid_loader)/batch_size)
    print('\n| Validation Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (x, y) in enumerate(valid_loader):
        x = x.view(-1, 3, resize, resize).repeat(args.num_samples, 1, 1, 1)
        y = y.repeat(args.num_samples)

        # move tensors to GPU if CUDA is available
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        
        with torch.no_grad() :
            x, y = Variable(x), Variable(y)

            if args.beta_type is "Blundell":
                beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
            elif args.beta_type is "Soenderby":
                beta = min(epoch / (num_epochs//4), 1)
            elif args.beta_type is "Standard":
                beta = 1 / m
            else:
                beta = 0
            
            # forward pass: compute predicted outputs by passing inputs to the model
            # output, loss_val = net(x, y, args.num_samples, batch_size, 10, "validation")
            output, kl_val = net(x)
            output = normalization_function(output)
            loss_val = vi(output, y, kl, beta)
            ll_val = loss_val.data.mean() - beta*kl_val.data
            kls_val.append(beta*kl_val.data)
            likelihoods_val.append(ll_val)
           
            # update average validation loss 
            valid_loss += loss_val.item()

            # preds = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            accuracy_val = (predicted.data.cpu() == y.cpu()).float().mean()
            # output = F.softmax(output, 1)
            results = torch.topk(output.cuda().data, k=1, dim=1)
            conf.append(results[0][0].item())
            total_val += y.size(0)
            average_loss = valid_loss/total_val

        # print training/validation statistics 
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] Average Validation Loss: %.4f Average Validation Accuracy: %.3f KL : %.4f Likelihood : %.4f' %(epoch, num_epochs, batch_idx+1,
                    len(valid_loader), average_loss , accuracy_val, sum(kls_val)/len(kls_val), sum(likelihoods_val)/len(likelihoods_val)))
        sys.stdout.flush()

    p_hat=np.array(conf)
    confidence_mean=np.mean(p_hat, axis=0)
    confidence_var=np.var(p_hat, axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
        
    # calculate average info
    print("\n| Final Training Accuracy : {:.3f} ; Final Validation Accuracy : {:.3f}".format(accuracy_train,accuracy_val))
    print("| Epistemic Uncertainity is : ", epistemic)
    print("| Aleatoric Uncertainity is : ", aleatoric)
    print("| Mean is : ", confidence_mean)
    print("| Variance is : ", confidence_var)
    
    if average_loss <= valid_loss_min:
        print('| Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,average_loss))
        state = {
                'net'  :net if use_cuda else net,
                'acc' : accuracy_val,
                'epoch' : epoch,
                'model_state' : net.state_dict()
        }
        if not os.path.isdir(args.save_folder + '/checkpoint'):
            os.mkdir(args.save_folder + '/checkpoint')
        save_point = args.save_folder + '/checkpoint/' + args.dataset + os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+file_name+'.t7')
        valid_loss_min = valid_loss

    diagnostics_to_write = {
        'Epoch': epoch, 
        'Loss': avg_train_loss, 
        'Accuracy': accuracy_train,
        "KL divergency" : sum(kls)/len(kls),
        "Log Likelihood" : sum(likelihoods)/len(likelihoods)
    }
    val_diagnostics_to_write = {
        'Validation Epoch': epoch, 
        'Loss': average_loss, 
        'Accuracy': accuracy_val,
        "KL divergency" : sum(kls_val)/len(kls_val),
        "Log Likelihood" : sum(likelihoods_val)/len(likelihoods_val)
    }
    values_to_write = {
        'Epoch':epoch, 
        'Confidence Mean: ':confidence_mean,
        'Confidence Variance:':confidence_var, 
        'Epistemic Uncertainity: ': epistemic, 
        'Aleatoric Uncertainity: ':aleatoric
    }
    with open(logfile, 'a') as lf:
        lf.write(str(diagnostics_to_write))

    with open(val_logfile, 'a') as lf:
        lf.write(str(val_diagnostics_to_write))

    with open(value_file, 'a') as lf:
        lf.write(str(values_to_write))

def normalization_function(x):
    return (x) / torch.sum(x, dim=0)

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train_and_val(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' %(cf.get_hms(elapsed_time)))
    print("===============================================================")
