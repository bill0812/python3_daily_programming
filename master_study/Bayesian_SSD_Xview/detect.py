# import basic packages
import torch,cv2, argparse, glob
import numpy as np
import os
import os.path as osp
from numpy import random
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

# import mine/reference packages
# from SSD_utils.model_Bayesian_Softplus import build_ssd, MultiBoxLoss
from SSD_utils.model_nonBayesian import build_ssd, MultiBoxLoss

from SSD_utils.data.xview import map_labels_contiguous, map_labels_name 
# from SSD_utils.data.utils import *
from SSD_utils.data.config import xview as cfg
from SSD_utils.data.config import HOME
from SSD_utils.data.augmentations import *

# note: if you used our download scripts, this should be right
XVIEW_ROOT = osp.join(HOME, "data/")

# sep up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_name, label_name_origin = map_labels_name(osp.join(XVIEW_ROOT, 'xview_labels.txt'))
print(label_name)

# Transforms
# resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
augument = Compose([
			ConvertFromInts(),
			ToAbsoluteCoords(),
			PhotometricDistort(),
			Expand((104, 117, 123)),
			RandomSampleCrop(),
			RandomMirror(),
			ToPercentCoords(),
			Resize(300),
			SubtractMeans((104, 117, 123))
		])

# detect area, then output to visualize
def detect(model, images, original_output, output, min_score, max_overlap, top_k):
    
    image_all = []
    image_original = []
    count = 1
    for img_name in images :
        img = cv2.imread(img_name)
        cv2.imwrite(original_output + "/" + str(count)+".jpg",img)
        image, _, _ = augument(img, np.array([[0.,0.,0.,0.]]), np.array([0]))
        # to rgb
        image = image[:, :, (2, 1, 0)]
        image_original.append(img)
        image_all.append(image)
        count += 1

    for index in range(len(image_all)):
        print("\r| Detecting [ {} / {} ] image ...".format(index+1,len(image_all)),end="")
        image_each = torch.from_numpy(image_all[index]).permute(2, 0, 1).to(device)

        # Forward prop.
        predicted_locs, predicted_scores, prior = model(image_each.unsqueeze(0))

        prior = prior.cuda()

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, prior, min_score=min_score,
                                                                max_overlap=max_overlap, top_k=top_k)

        # Decode class integer labels
        det_labels = [label_name[l] for l in det_labels[0].to(device).tolist()]
        # print(det_labels)

        # set font 
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
        if det_labels == ['__NOCLASS__']:

            # Just save original image
            cv2.imwrite(output + "/" + str(index+1)+".jpg",image_original[index])

        else :
            # Annotate
            # print(det_boxes)
            for i in range(det_boxes[0].size(0)):

                # Boxes
                box_location = det_boxes[0][i].tolist()
                
                xmin, ymin, xmax, ymax = box_location
                cv2.putText(image_original[index], det_labels[i],(int(xmin*400), int(ymin*400-10)), font, 0.5, (0,255,0), 1)
                image = cv2.rectangle(image_original[index],(int(xmin*400), int(ymin*400)),(int(xmax*400), int(ymax*400)), (0,255,0), 1)
                
            cv2.imwrite(output + "/" + str(index+1)+".jpg",image_original[index])

# for chipping images
def chip_image(img_name,shape=(400,400)):
        """
        Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
            multiple chips are clipped: each portion that is in a chip is labeled. For example,
            half a building will be labeled if it is cut off in a chip. If there are no boxes,
            the boxes array will be [[0,0,0,0]] and classes [0].
            Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.
        Args:
            img: the image to be chipped in array format
            coords: an (N,4) array of bounding box coordinates for that image
            classes: an (N,1) array of classes for each bounding box
            shape: an (W,H) tuple indicating width and height of chips
        Output:
            An image array of shape (M,W,H,C), where M is the number of chips,
            W and H are the dimensions of the image, and C is the number of color
            channels.  Also returns boxes and classes dictionaries for each corresponding chip.
        """
        img = cv2.imread(img_name)
        # print(img)
        height,width,_ = img.shape
        wn,hn = shape
        assert wn == hn
        
        # get the most left pixel left
        # stride times for width and height
        stride = wn//2
        w_left, h_left = width%stride, height%stride
        w_max, h_max = (width - w_left) - stride, (height - h_left) - stride
       
        # store images, boxes, and classes
        images = {}

        # chipped images using stride
        k = 0
        for i in range(0,w_max,stride):
            for j in range(0,h_max,stride) :
                result = check_chip(np.array(img[j:(j+hn),i:(i+wn),:3]))
                if result :
                    chip = np.array(img[j:(j+hn),i:(i+wn),:3])
                    images[k] = chip.astype(np.uint8)
                    cv2.imwrite("../detection_result_original/" + str(k)+".jpg", images[k])
                    k = k+1

        return images

# check image for whether there is all black
def check_chip(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find frequency of pixels in range 0-255 
    histr = cv2.calcHist([gray],[0],None,[256],[0,256]) 

    # percent check
    sum_all = np.sum(histr)
    black_one = histr[0]
    if black_one/sum_all >= 0.5 :
        return False
    else :
        return True

# find only cars object image
def find_object(filename,output) :

    result = []
    count = 1
    low_bound = 5000
    up_bound = 5201
    # # set font 
    font = cv2.FONT_HERSHEY_SIMPLEX
    # retrieve data from ground truth
    with open(filename,"r") as file_detail :
        for line, row in enumerate(file_detail) :
            
            # retrieve xmin, ymin, xmax, ymax, class_name
            img_name, xmin, ymin, xmax, ymax, class_name = row.split(",")
            xmin = int(float(xmin))
            ymin = int(float(ymin))
            xmax = int(float(xmax))
            ymax = int(float(ymax))
            class_name = int(float(class_name.split("\n")[0]))
        
            if (xmin, ymin, xmax, ymax, class_name) == ('', '', '', '', '') :
                continue

            if class_name == 73 :
                if img_name in result :
                    pass
                else :
                    if low_bound < count < up_bound :
                        result.append(img_name)
                    count += 1 
    
    return result
# main function
if __name__ == '__main__':
    
    test_data_img = "/media/bill/BillData/比爾Bayesian-Satillite /dataset/Xview/dataset_400_new/validation_original/27"
    # train_data_img = "/media/bill/BillData/比爾Bayesian-Satillite /dataset/Xview/dataset_400_new/training_data.csv"
    # val_dtat_img = "/media/bill/BillData/比爾Bayesian-Satillite /dataset/Xview/dataset_400_new/validation_data.csv"
    checkpoint = "../cars_buildings/outputs/XVIEW_SSD_new.pt"

    parser = argparse.ArgumentParser() 
    parser.add_argument("--tes_image_dir", default=test_data_img,
                        help="Path to folder containing image chips")
    # parser.add_argument("--chip_index", default="542", type=int,
    #                     help="Output specific image's output")
    # parser.add_argument("--chip_size", default="400", type=int,
    #                     help="Chipped to specific size")
    
    parser.add_argument("--output_visualize", default="../detection_result", type=str,
                        help="Chipped to specific size")
    parser.add_argument("--output_original", default="../detection_original", type=str,
                        help="Chipped to specific size")
    parser.add_argument("--output_label", default="../detection_labeled", type=str,
                        help="Chipped to specific size")
    args = parser.parse_args()

    # Load model checkpoint
    print('| Loading checkpoint ...')
    checkpoint_dict = torch.load(checkpoint)
    print("| Last loss so far is {:.3f}.".format(checkpoint_dict["loss"]))
    print("| Best Average loss so far is {:.3f}.".format(checkpoint_dict["best_loss"]))
    print("====================================================")
    
    print(cfg['num_classes'])
    model = build_ssd(cfg['min_dim'], cfg['num_classes'])
    model.load_weights(checkpoint,checkpoint_dict)
    model = model.to(device)
    model.eval()

    # ==============================
    # preparing image
    print("| Lodaing Data ...")

    file_names = glob.glob(test_data_img + "/*.jpg")
    file_names.sort()
    # if not os.path.exists(args.output_label) :
    #     os.makedirs(args.output_label)

    # file_names = find_object(train_data_img,args.output_label)

    print("| Start Detecting ...")

    if not os.path.exists(args.output_visualize):
        os.makedirs(args.output_visualize)
    if not os.path.exists(args.output_original) :
        os.makedirs(args.output_original)

    detect(model, file_names, args.output_original, args.output_visualize, 0.05, 0.5, 20)

    print("\n| Finish ... Done ! !")
