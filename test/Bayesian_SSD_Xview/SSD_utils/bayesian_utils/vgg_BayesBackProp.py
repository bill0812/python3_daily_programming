'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from BackProp.BNNLayer import BNNLayer

# setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            BNNLayer(512, 4096),
            BNNLayer(4096, 10),
        )

    def forward(self, true_x, y, num_samples, batch_size, classes, mode):

        if mode == "train" or  mode == "validation" :
            outputs = torch.zeros(num_samples, batch_size, classes).to(DEVICE)
            log_priors = torch.zeros(num_samples).to(DEVICE)
            log_variational_posteriors = torch.zeros(num_samples).to(DEVICE)
            for sample in range(num_samples) :
                for layer in self.features :
                    if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d):
                        outputs[sample] = layer(outputs[sample])
                    else :
                        outputs[sample] = layer.foward(outputs[sample])
                        log_priors[sample] += layer.log_prior
                        log_variational_posteriors[sample] += layer.log_variational_posterior
            
                outputs[sample] = outputs[sample].view(outputs[sample].size(0), -1)

                for layer in self.classifier :
                    outputs[sample] = layer.foward(outputs[sample])
                    log_priors[sample] += layer.log_prior
                    log_variational_posteriors[sample] += layer.log_variational_posterior

            # the mean of samples of ouput 
            output_result = outputs.mean()
            # the mean of prior probability
            log_prior = log_priors.mean()
            # the mean of posterior probability
            log_variational_posterior = log_variational_posteriors.mean()

            # calculate likelihood
            negative_log_likelihood = F.nll_loss(outputs.mean(0), y, size_average=False)

            # kl divergence : KL[ q(w|theta) | P(w)] - likelihood ( log(P(D|w)) )
            loss = (log_variational_posterior - log_prior)/batch_size + negative_log_likelihood

            return output_result, loss

        elif mode == "test" :
            for layer in self.features :
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d):
                    x = layer(x)
                else :
                    x = layer.foward(x)
        
            x = x.view(x.size(0), -1)

            for layer in self.classifier :
                x = layer.foward(x)

            return x

    @staticmethod
    def loss_fn(kl, n_batch):
        return (kl / n_batch).mean()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    
    for v in cfg:
        if batch_norm:
            conv2d = BNNLayer(in_channels, v)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            conv2d = BNNLayer(in_channels, v)
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    return nn.Sequential(*layers)
    
# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
#           512, 512, 512, 512, 'M'],
# }

cfg = {
    'A': [64, 128, 256, 512, 512],
    'B': [64, 128, 256, 512, 512],
    'D': [64, 128, 256, 512, 512],
    'E': [64, 128, 256, 512, 512],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
