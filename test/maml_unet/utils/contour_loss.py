import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import cv2

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class Contour_loss(torch.nn.Module):
    def __init__(self, window_size = 5, K = 5):
        super(Contour_loss, self).__init__()
        
        # edge map
        self.window_size = window_size
        self.element = cv2.getStructuringElement(cv2.MORPH_RECT,(self.window_size, self.window_size))
        
        # gaussian weighting
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, gt_image):
        
        # check channel
        if gt_image.is_cuda:
            gt_image = gt_image.cpu().numpy()
        
        (batch_size, channel, _, _) = gt_image.shape

        batch_M_weight_map = None
        for batch_idx in range(batch_size):
            M_weight_map = c_loss(gt_image[batch_idx, :, :, :], window_size = 5, K = 5)
            
            if batch_idx == 0:
                batch_M_weight_map = M_weight_map
            else:
                batch_M_weight_map = torch.cat((batch_M_weight_map, M_weight_map), 0)

        batch_M_weight_map = batch_M_weight_map.transpose(1, 3)

        return batch_M_weight_map

def c_loss(gt_image, window_size = 5, K = 5):

#     gt_image = gt_image.numpy()
    # print(gt_image.shape)
    gt_image = np.rollaxis(gt_image, 0, 2)
    (channel, _, _) = gt_image.shape
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(window_size, window_size))
    dilated_image = cv2.dilate(gt_image, element)
    eroded_image = cv2.erode(gt_image, element)
    edge_map = K*cv2.absdiff(dilated_image,eroded_image) # or not diff
    edge_map = torch.FloatTensor(np.rollaxis(edge_map, 2)).unsqueeze(0).cuda()
    
    window = create_window(window_size, channel)
    
    edge_map = edge_map.cuda()
    if edge_map.is_cuda:
        window = window.cuda(edge_map.get_device())
    
    M_weight_map = F.conv2d(edge_map, window, padding = window_size//2, groups = channel)

    return M_weight_map



