# import basic packages
import sys, cv2, glob, json, argparse, os, csv
import numpy as np
from tqdm import tqdm

# import pytorch stuff
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from torch.autograd import Variable

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class image_dataset(Dataset):
    def __init__(self, images_filename=None, boxes_filename=None, classes_filename=None, chip_dir_name=None, is_chipped=True, batch_size=None):
        
        # get chipped image dir and chipped count in dir
        self.chip_dir = chip_dir_name
        self.count = 0
        self.dir_count = 0
        self.batch_size = batch_size
        self.boxes_filename = boxes_filename
        self.label_txt = "/home/bill/Desktop/research_bayesian_bigscale/Bayesian_SSD_Xview/SSD/SSD_utils/data/xview_labels.txt"
        self.label_name_origin = dict()
        labels_index = open(self.label_txt, 'r')
        for line in labels_index:
            ids = line.split(',')
            self.label_name_origin[int(ids[0])] = ids[2].split('\n')[0]

        # deal with label file and chipping and find each iter and for loop the iterations
        file_names = glob.glob(images_filename + "*.tif")
        file_names.sort()
        self.iter_time = (len(file_names)//batch_size)
        self.last_iter_patch = len(file_names)%batch_size

        # get labels
        self.coords, self.chips , self.classes = self.get_labels(boxes_filename)

        # print info
        print("| Total Tif Images are : {}".format(len(file_names)))
        print("| Batch Size is : {}".format(batch_size))
        print("| Total Iterations are : {}".format(self.iter_time+1))
        print("| We are choosing { Buildings and small cars !!")
        print("| Start Processing ...\n")
        # start to retrive images and store
        for each_iter in range(self.iter_time) :
            start = each_iter * batch_size
            end = start + batch_size
            iter_image = file_names[start:end]
            print("\r| iter [ %d / %d ]" %(each_iter+1,self.iter_time+1),end="")
            self.retrieve_data(iter_image,each_iter+1)
            sys.stdout.flush()

        iter_image = file_names[len(file_names)-self.last_iter_patch:]
        print("\r| iter [ %d / %d ]" %(self.iter_time+1,self.iter_time+1),end="")
        self.retrieve_data(iter_image,1)
        sys.stdout.flush()
        
        print("| Finishing Chipping and Creating Dataset !!")

    # iter to retrieve data by batch size and save
    def retrieve_data(self,iter_image,iter) :

        images_name, ground_truth = [], []
        print("\n| Processing  {} Iter ... ".format(iter))
        for filename in tqdm(iter_image):
            img = np.array(cv2.imread(filename))
            img_name = filename.split("/")[-1]
            
            img_coords = self.coords[self.chips == img_name]
            img_classes = self.classes[self.chips == img_name]

            images_name_each , ground_truth_each = self.chip_image(img_name, img, img_coords, img_classes)
            
            images_name.extend(list(images_name_each))
            ground_truth.extend(list(ground_truth_each))

        # length should be the same
        assert len(images_name) == len(ground_truth)
        
        with open(self.chip_dir + "/training_data.csv","a") as train :
            writer = csv.writer(train)
            for each_file in range(len(images_name)) :
                for each_gt in range(len(ground_truth[each_file])) :
                    
                    transfer_data = list(ground_truth[each_file][each_gt])
                    xmin = transfer_data[0][0]
                    ymin = transfer_data[0][1]
                    xmax = transfer_data[0][2]
                    ymax = transfer_data[0][3]
                    writer.writerow([images_name[each_file],xmin,ymin,xmax,ymax,transfer_data[1]])

    # get labels from json file
    def get_labels(self,fname):
        """
        Gets label data from a geojson label file
        Args:
            fname: file path to an xView geojson label file
        Output:
            Returns three arrays: coords, chips, and classes corresponding to the
                coordinates, file-names, and classes for each ground truth.
        """
        with open(fname) as f:
            data = json.load(f)

        coords = np.zeros((len(data['features']),4))
        chips = np.zeros((len(data['features'])),dtype="object")
        classes = np.zeros((len(data['features'])))

        for i in tqdm(range(len(data['features']))):
            if data['features'][i]['properties']['bounds_imcoords'] != []:
                b_id = data['features'][i]['properties']['image_id']
                val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
                chips[i] = b_id
                classes[i] = data['features'][i]['properties']['type_id']
                if val.shape[0] != 4:
                    print("Issues at %d!" % i)
                else:
                    coords[i] = val
            else:
                chips[i] = 'None'

        return coords, chips, classes

    # check image for whether there is all black
    def check_chip(self,image) :
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

    def chip_image(self,img_name,img,coords,classes,shape=(400,400)):
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
        images_name = {}
        total_groundtruth = {}

        label_name_origin = self.label_name_origin
        chip_dir = self.chip_dir

        # chipped images using stride
        k = 0
        for i in range(0,w_max,stride):
            for j in range(0,h_max,stride) :

                result = self.check_chip(np.array(img[j:(j+hn),i:(i+wn),:3]))

                if result :
                    index = 0
                    ground_truth = list()
                    for each_coord in (coords) :
                        
                        if classes[index] == 73 or classes[index] == 18 :
                            # if (int(each_coord[0]) >=0) and (int(each_coord[1]) >= 0) and (int(each_coord[2] )>= 0) and (int(each_coord[3]) >= 0) :
                            x = np.logical_or( np.logical_and((each_coord[0]<(i+wn)),(each_coord[0]>i)),
                                            np.logical_and((each_coord[2]<(i+wn)),(each_coord[2]> i)))
                            
                            y = np.logical_or( np.logical_and((each_coord[1]<(j+hn)),(each_coord[1]>j)),
                                            np.logical_and((each_coord[3]<(j+hn)),(each_coord[3]> j)))
                            # print(x,y)
                            if x or y :

                                possible_out = [np.clip(each_coord[0]-(i),0,wn),
                                                            np.clip(each_coord[1]-(j),0,hn),
                                                            np.clip(each_coord[2]-(i),0,wn),
                                                            np.clip(each_coord[3]-(j),0,hn)]
                                
                                if (abs(possible_out[2]-possible_out[0])*abs(possible_out[3]-possible_out[1]) >\
                                        abs(each_coord[2]-each_coord[0])*abs(each_coord[3]-each_coord[1])*0.7) :
                                    
                                    ground_truth.append([np.array(possible_out),classes[index]])

                        index += 1
                    if (len(ground_truth)) != 0:
                        total_groundtruth[k] = np.array(ground_truth)
                        chip = np.array(img[j:(j+hn),i:(i+wn),:3])
                        images[k] = chip.astype(np.uint8)
                        images_name[k] = self.save_images_to_dir(images[k],total_groundtruth[k][:,0],total_groundtruth[k][:,1],self.count+1,label_name_origin, chip_dir,self.dir_count+1)

                        k += 1
                        self.count += 1
        
        images_name_np = np.array(list(images_name.values()))
        ground_truth_np = np.array(list(total_groundtruth.values()))
        return images_name_np, ground_truth_np

    def save_images_to_dir(self, image, boxes, class_name, count, label_name_origin, chip_dir, dir_count) :
        
        # how many boxes you have, then how many classes you get
        assert len(boxes) == len(class_name)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if not os.path.exists(chip_dir + "/training_original/" + str(dir_count)):
            os.makedirs(chip_dir + "/training_original/" + str(dir_count)) 

        if not os.path.exists(chip_dir + "/training_labeled/" + str(dir_count)):
            os.makedirs(chip_dir + "/training_labeled/" + str(dir_count)) 

        cv2.imwrite(chip_dir + "/training_original/" + str(dir_count)  + "/chipped_images_with_boundingbox_" + str(count) + ".jpg",image)
        
        # draw chipped images
        for index in range(len(boxes)) :
            xmin, ymin, xmax, ymax = boxes[index]
            if class_name[index] not in label_name_origin.keys() :
                class_name[index] = 0
            cv2.putText(image, label_name_origin[class_name[index]],(int(xmin), int(ymin-10)), font, 0.5, (0,255,255), 1)
            image = cv2.rectangle(image,(int(xmin), int(ymin)),(int(xmax), int(ymax)), (0,255,255), 1)
        
        cv2.imwrite(chip_dir + "/training_labeled/" + str(dir_count)  + "/chipped_images_with_boundingbox_" + str(count) + ".jpg",image)
        
        if count%100 == 0 :
            self.dir_count += 1

        return chip_dir + "/training_original/" + str(dir_count)  + "/chipped_images_with_boundingbox_" + str(count) + ".jpg"

def plot_image(file_names, start, end, place, img_dir):
    
    result = {}
    # get mapping data
    label_name_origin, count_dict = dict(), dict()
    label_txt = "/home/bill/Desktop/research_bayesian_bigscale/Bayesian_SSD_Xview/SSD/SSD_utils/data/xview_labels.txt"
    labels_index = open(label_txt, 'r')
    for line in labels_index:
        ids = line.split(',')
        label_name_origin[int(ids[0])] = ids[2].split('\n')[0]
        count_dict[ids[2].split('\n')[0]] = 0

    class_dict = dict()
    class_dict["buildings"] = 0
    class_dict["small cars"] = 0

    # retrieve data from ground truth
    with open(file_names,"r") as file_detail :
        for line, row in enumerate(file_detail) :
            
            # retrieve xmin, ymin, xmax, ymax, class_name
            img_name, xmin, ymin, xmax, ymax, class_name = row.split(",")
            # print((class_name))
            # xmin = int(float(xmin))
            # xmax = int(float(xmax))
            # ymax = int(float(ymax))
            class_name = int(float(class_name.split("\n")[0]))
           
            # if img_name not in result :
            #     result[img_name] = []

            if (xmin, ymin, xmax, ymax, class_name) == ('', '', '', '', '') :
                continue

            # result[img_name].append(np.array([xmin, ymin, xmax, ymax, class_name]))

            # if class_name == 18 :
            #     class_dict["small cars"] += 1
            # elif class_name == 73 :
            #     class_dict["buildings"] += 1

            count_dict[label_name_origin[class_name]] += 1
    
    return count_dict #class_dict

    # # =======================================
    # # draw chipped images
    # count = 1
    # image_item = 0
    # for name, data in result.items() :
        
    #     if count <= (end-start) and start <= image_item < end :
    #         print("\r| Processing [ %d / %d ] Images" %(count,(end-start)),end="")
    #         sys.stdout.flush()
            
            # data = np.array(data)
            # # how many boxes you have, then how many classes you have, too
            # assert data[:,0:4].shape[0] == data[:,4:5].shape[0]

            # # set font 
            # font = cv2.FONT_HERSHEY_SIMPLEX

            # # read image
            # image = cv2.imread(name)

            # cv2.imwrite(place + str(count) + "_original.jpg",image)

            # # draw chipped images
            # for index_box in data :
            #     xmin, ymin, xmax, ymax, class_name = index_box
            #     if class_name not in label_name_origin.keys() :
            #         class_name = 0
            #     cv2.putText(image, label_name_origin[class_name],(xmin, ymin-10), font, 0.5, (0,255,255), 1)
            #     image = cv2.rectangle(image,(xmin, ymin),(xmax, ymax), (0,255,255), 1)
            
            # cv2.imwrite(place + str(count) + "_label.jpg",image)
            # count += 1

        # image_item += 1
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir_name", default="/media/bill/BillData/比爾Bayesian-Satillite /dataset/Xview/train_images/",
                        help="Path to folder containing image chips \
                        (ie 'xview/train_images/' ")
    parser.add_argument("--json_file_path", default="/media/bill/BillData/比爾Bayesian-Satillite /dataset/Xview/xView_train.geojson",
                        help="File path to GEOJSON coordinate file")
    parser.add_argument("--chip_image_dir_name", default="/media/bill/BillData/比爾Bayesian-Satillite /dataset/Xview/dataset_400_new",
                        help="File path to chipped image files")
    parser.add_argument("--batch_size", default="256", type=int,
                        help="Batch Size for Dealing with data every single time")
    parser.add_argument("--visualize_start", default="0", type=int,
                        help="Output visualization Data in Desktop")
    parser.add_argument("--visualize_end", default="200", type=int,
                        help="Output visualization Data in Desktop")
    
    args = parser.parse_args()

    # print("| Chipping Images From Directory : \n{}".format(args.image_dir_name))
    # print("| Using Label File : \n{}".format(args.json_file_path))
    # print("===============================================================")

    # if not os.path.exists(args.chip_image_dir_name):
    #     os.makedirs(args.chip_image_dir_name)

    # print("Saving Data to : {}\n( Including Images files )".format(args.chip_image_dir_name))
    # print("==================================================")

    # dataset = image_dataset(args.image_dir_name, args.json_file_path,
    #             chip_dir_name=args.chip_image_dir_name, batch_size=args.batch_size)

    # print("| Visualization ...")
    # print("======================")

    # # visualization
    place = "/home/bill/Desktop/visualization_training_1/"

    # deal with label file and chipping and find each iter and for loop the iterations
    file_names = glob.glob(args.chip_image_dir_name + "/validation_data.csv")
    file_names.sort()

    print("| Processing {}".format(file_names[0]))
    print("=====================================")

    # create directory if not exist
    # if not os.path.exists(place):
    #     os.makedirs(place)

    print(plot_image(file_names[0],args.visualize_start,args.visualize_end,place,args.chip_image_dir_name))
    # print("| Finishing Visualization !!")