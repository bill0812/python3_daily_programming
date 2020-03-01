# import basic packages
import sys

# import pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# import mine/ reference packages
from function.box_utils import *
from function.l2norm import *
from function.prior_box import *
from data.utils import *

# import Bayes by BackProp
from bayesian_utils.BackProp.BNN import BNN
from bayesian_utils.BackProp.BNNLayer import BNNLayer, BNNFC_Layer

# device for cuda or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SSD300 CONFIGS
xview = {
	'num_classes': 61,
	'lr_steps': (300, 600, 900, 1200, 1500, 1800),
	'max_iter': 1800,
	'feature_maps': [38, 19, 10, 5, 3, 1],
	'min_dim': 300,
	'steps': [8, 16, 32, 64, 100, 300],
	'min_sizes': [21, 45, 99, 153, 207, 261],
	'max_sizes': [45, 99, 153, 207, 261, 315],
	'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
	'variance': [0.1, 0.2],
	'clip': True,
	'name': 'XVIEW',
}

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}
# =============================================================
class SSD(nn.Module):
    
    # basic setup
    def __init__(self, size, base, extras, head, num_classes, num_samples):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.cfg = xview
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), requires_grad=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
    
    # forward propagation
    def forward(self, x, mode):

        assert mode in {'train', 'validation', 'test'}, 'BNNLayer Mode Not Found'
        
        sources = list()
        loc = [for i in range(self.num_samples)]
        conf = [for i in range(self.num_samples)]
        loc_final = list()
        conf_final = list()
        total_kl, total_l, total_c = 0., 0. ,0.

        for i in range(self.num_samples) :


            # apply vgg up to conv4_3 relu
            for k in range(23):
                if mode == "train" or mode =="validation":
                    x, net_kl = self.vgg.forward(x, "forward")
                    total_kl += net_kl
                else :
                    x = self.vgg.foward(x, mode='MC')
            
            s = self.L2Norm(x)
            sources.append(s)

            # here are for append each layer 
            # including vgg_base, extra, multibox
            # ======================
            # apply vgg up to fc7
            for k in range(23, len(self.vgg)):
                if mode == "train"  or mode =="validation":
                    x, net_kl = self.vgg.foward(x, "forward")
                    total_kl += net_kl
                else :
                    x = self.vgg.forward(x, mode="MC")

            sources.append(x)
            # apply extra layers and cache source layer outputs
            for k, v in enumerate(self.extras):

                if mode == "train" or mode =="validation":
                    x, net_kl = v.foward(x, "forward")
                    x = F.relu(x, inplace=True)
                    total_kl += net_kl
                else :
                    x = v.forward(x, mode="MC")
                    x = F.relu(x, inplace=True)

                if k % 2 == 1:
                    sources.append(x)
            
            # apply multibox head to source layers
            for (x, l, c) in zip(sources, self.loc, self.conf):

                if mode == "train"  or mode =="validation":
                    l_x, net_kl_l = l.foward(x, "forward")
                    c_x, net_kl_c = c.foward(x, "forward")
                    total_kl += net_kl_l
                    total_kl += net_kl_c
                else :
                    l_x = l.forward(x, mode="MC")
                    c_x = l.forward(x, mode="MC")

                loc[i].append(l_x.permute(0, 2, 3, 1).contiguous())
                conf[i].append(c_x.permute(0, 2, 3, 1).contiguous())
        
        loc_final = torch.mean(loc, 0)
        conf_final = torch.mean(conf, 0)
        # ======================

        # cat all location prediction / classification scores
        loc_final = torch.cat([o.view(o.size(0), -1) for o in loc_final], 1)
        conf_final = torch.cat([o.view(o.size(0), -1) for o in conf_final], 1)

        # loc preds
        loc_pred = loc_final.view(loc.size(0), -1, 4)
        class_pred = conf_final.view(conf.size(0), -1, self.num_classes)

        if mode=="train" or mode =="validation":

            return loc_pred, class_pred, self.priors, total_kl
        else :
            return loc_pred, class_pred, self.priors

    # loading weights if checkpoint specify
    def load_weights(self, base_file, check_point_dict):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth' or '.pt':
            print('\n| Loading weights into state dict...')
            self.load_state_dict(check_point_dict["model_state_dict"])
            print('| Finished!')
        else:
            print('| Error => Sorry only .pth, .pkl and .pt files supported.')

    # detect object after prediction
    def detect_objects(self, predicted_locs, predicted_scores, prior_data, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = prior_data.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], prior_data))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.num_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

# This function is derived from torchvision VGG make_layers()
# # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# here for vgg based nerwork
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            if batch_norm:
                conv2d = BNNLayer(in_channels, v, kernel_size=3, padding=1, activation='bn2d_relu', , prior_mean=0, prior_rho=1)
                layers += [conv2d]
            else:
                conv2d = BNNLayer(in_channels, v, kernel_size=3, padding=1, activation='relu', , prior_mean=0, prior_rho=1)
                layers += [conv2d]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = BNNLayer(512, 1024, kernel_size=3, padding=6, dilation=6, activation="relu", prior_mean=0, prior_rho=1)
    conv7 = BNNLayer(1024, 1024, kernel_size=1, activation="relu", prior_mean=0, prior_rho=1)
    layers += [pool5, conv6, conv7]
    return layers

#  here for extra layers
def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [BNNLayer(in_channels, cfg[k + 1],
                    kernel_size=(1, 3)[flag], stride=2, padding=1, activation='none', prior_mean=0, prior_rho=1)]
            else:
                layers += [BNNLayer(in_channels, v,
                    kernel_size=(1, 3)[flag], activation='none', prior_mean=0, prior_rho=1)]
            flag = not flag
        in_channels = v

    return layers

# here are multibox
def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]

    for k, v in enumerate(vgg_source):
        loc_layers += [BNNLayer(vgg[v].out_channels,
                    cfg[k] * 4, kernel_size=3, padding=1, activation='none', prior_mean=0, prior_rho=1)]
        conf_layers += [BNNLayer(vgg[v].out_channels,
                    cfg[k] * num_classes, kernel_size=3, padding=1, activation='none', prior_mean=0, prior_rho=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        
        loc_layers += [BNNLayer(v.out_channels,
                    cfg[k] * 4, kernel_size=3, padding=1, activation='none', prior_mean=0, prior_rho=1)]
        conf_layers += [BNNLayer(v.out_channels,
                    cfg[k] * num_classes, kernel_size=3, padding=1, activation='none', prior_mean=0, prior_rho=1)]
    vgg = BNN(vgg)
    extra_layers = BNN(extra_layers)
    loc_layers = BNN(loc_layers)
    conf_layers = BNN(conf_layers)

    return vgg, extra_layers, (loc_layers, conf_layers)

# building ssd network
def build_ssd(size=300, num_classes=21):

    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(size, base_, extras_, head_, num_classes)

# here are loss function
class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = 0
        self.priors_xy = 0
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, prior_data, boxes, labels, total_kl, batch_size):
        """
        Forward propagation.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        self.priors_cxcy = prior_data
        self.priors_xy = cxcy_to_xy(prior_data)

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)
            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)


            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)
        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)


        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)


        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss , self.alpha * loc_loss, (total_kl / n_batch).mean()