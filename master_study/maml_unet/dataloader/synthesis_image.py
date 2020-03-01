import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import cv2
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import glob

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def img_loader(path, resize):
    img_pil =  Image.open(path)
    preprocess_img = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.ToTensor()
    ])
    img_tensor = preprocess_img(img_pil)
    return img_tensor

def load_mat(reflection_target, str_tag):
    
    try:
        reflection_target = hdf5storage.loadmat(reflection_target)
        reflection_target = reflection_target[str_tag]
        preprocess_reflection_target = np.asarray(reflection_target, np.float32)
        preprocess_reflection_target = preprocess_reflection_target / 5.0
        preprocess_reflection_target = transform.resize(preprocess_reflection_target, (512, 512))
        preprocess_reflection_target = preprocess_reflection_target * 5.0
        preprocess_reflection_target = preprocess_reflection_target[np.newaxis, :, :]
    except:
        preprocess_reflection_target = np.array([[]], np.float32)
        print(reflection_target)
    return torch.from_numpy(preprocess_reflection_target) # preprocess_reflection_target 

def generate_dark_image(img, true_masks):
    
    dark_image = torch.zeros(img.size())
    dark_image[0,:,:] = img[0,:,:] * true_masks.type(torch.FloatTensor)
    dark_image[1,:,:] = img[1,:,:] * true_masks.type(torch.FloatTensor) 
    dark_image[2,:,:] = img[2,:,:] * true_masks.type(torch.FloatTensor)
    return dark_image


class Synthesis_Image(Dataset):
    

    def __init__(self, root, mode, n_way=None, k_shot=None, k_query=None, batchsz=None, resize=512, mat=False, startidx=0, test_data=None):
        """
        root :
        |- input/ includes all classes
        |- - class_1/ includes all of images
        |- - class_2/ includes all of images
        |- - ... includes all of images
        |- ground_truth/
        
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """
        
        self.mode = mode
        if self.mode == "train" or self.mode == "validation" :
            self.batchsz = batchsz  # batch of set, not batch of imgs
            self.n_way = n_way  # n-way
            self.k_shot = k_shot  # k-shot
            self.k_query = k_query  # for evaluation
            self.setsz = self.n_way * self.k_shot  # num of samples per set
            self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
            self.resize = resize  # resize to
            self.startidx = startidx  # index label not from 0, but from startidx
            self.mat = mat
            print('INFO: Shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
            mode, batchsz, n_way, k_shot, k_query, resize))

            # set up path
            if mode == "train" :
                imgdir_root = os.path.join(root, 'train')
            elif mode == "validation" :
                imgdir_root= os.path.join(root, 'validation')
            else :
                raise ValueError("Wrong Data Directory !")

            # get image path
            self.input_path = os.path.join(imgdir_root, 'input')  # image path
            self.gt_path = os.path.join(imgdir_root, 'ground_truth')  # ground path

            # load all images filename
            self.input_data, self.gt_data, self.mat_data = [], [], []
            input_filename, gt_filename, mat_filename = self.loadfile(self.input_path, self.gt_path) 
            # print(input_filename.items())
            for i, (k, v) in enumerate(input_filename.items()):
                self.input_data.append(v)  # [[input1, input2, ...], [input111, ...]]
            for i, (k, v) in enumerate(gt_filename.items()):
                self.gt_data.append(v)  # [[gt1, gt2, ...], [gt111, ...]]

            if self.mat :
                print("Using Mat to rebuild sun image")
                for i, (k, v) in enumerate(mat_filename.items()):
                    self.mat_data.append(v)  # [[mat1, mat2, ...], [mat111, ...]]

            # get class number
            self.cls_num = len(self.input_data)
            self.create_batch(self.batchsz)
        elif self.mode == "test" :
            self.batchsz = batchsz
            self.resize = resize  # resize to
            self.test_data = test_data
            self.xname = []
            self.yname = []
            if self.test_data == "test" :
                imgdir_root = os.path.join(root, 'test/')
                # get image path
                self.input_path = os.path.join(imgdir_root, 'input_3/')  
                self.gt_path = os.path.join(imgdir_root, 'ground_truth') 
                image = sorted(glob.glob(self.input_path + "*.png"))
                for i, path in enumerate(image):
                    directory = os.path.dirname(path)
                    basename = os.path.basename(path)
                    self.xname.append(path)
                    self.yname.append(self.gt_path + directory[-2:] + "/" + "ground_truth_" + basename[6:])
            elif self.test_data == "val":
                imgdir_root = os.path.join(root, 'validation/')
                self.input_path = os.path.join(imgdir_root, 'input/class_*/input/')  
                self.gt_path = os.path.join(imgdir_root, 'ground_truth/')  # ground path
                image = sorted(glob.glob(self.input_path + "*.jpg"))
                for i, path in enumerate(image):
                    basename = os.path.basename(path)
                    self.xname.append(path)
                    self.yname.append(self.gt_path + basename[:-6] + ".jpg")
            else :
                raise ValueError("Wrong Data Directory !")
            assert len(self.xname) == len(self.yname)
        elif self.mode == 'train_normal' :
            self.batchsz = batchsz
            self.resize = resize  # resize to
            imgdir_root = os.path.join(root, 'train/')
            
            # get image path
            self.input_path = os.path.join(imgdir_root, 'input/class_*/input/')  
            self.gt_path = os.path.join(imgdir_root, 'ground_truth/')  # ground path
            image = sorted(glob.glob(self.input_path + "*.jpg"))
        
            self.xname = []
            self.yname = []
            for i, path in enumerate(image):    
                basename = os.path.basename(path)
                self.xname.append(path)
                self.yname.append(self.gt_path + basename[:-6] + ".jpg")
            assert len(self.xname) == len(self.yname)
        elif self.mode == 'val_normal' :
            self.batchsz = batchsz
            self.resize = resize  # resize to
            imgdir_root = os.path.join(root, 'validation/')
            
            # get image path
            self.input_path = os.path.join(imgdir_root, 'input/class_*/input/')  
            self.gt_path = os.path.join(imgdir_root, 'ground_truth/')  # ground path
            image = sorted(glob.glob(self.input_path + "*.jpg"))

            self.xname = []
            self.yname = []
            for i, path in enumerate(image):
                basename = os.path.basename(path)
                self.xname.append(path)
                self.yname.append(self.gt_path + basename[:-6] + ".jpg")

            assert len(self.xname) == len(self.yname)
        else :
            raise ValueError("Wrong Data Directory !")

    def loadfile(self, input_path, gt_path):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        
        # store for dict input dict
        input_dict = {}
        gt_dict = {}
        mat_dict = {}
        
        # get all class input images, then store in dict
        input_dir = []
        mat_dir = []
        for dir_walk in os.walk(input_path) :
            
            if os.path.basename(os.path.normpath(dir_walk[0])) == "input" and len(dir_walk[1]) != 0:
                for each_file in dir_walk[1] :
                    if each_file == ".reserved" :
                        continue
                    input_dir.append(os.path.join(dir_walk[0] , each_file) + "/input/")
                    if self.mat :
                        mat_dir.append(os.path.join(dir_walk[0] , each_file) + "/mat_mask/")
                
        # assertion
        if self.mat :
            assert len(mat_dir) == len(input_dir)
        
        # save to dict
        for class_idx, each_dir in enumerate(input_dir) :
            input_image = glob.glob(os.path.join(each_dir, '*.jpg'))
            if self.mat :
                mat_mask = glob.glob(os.path.join(each_dir, '*.mat'))
                 
                # assertion
                assert len(input_image) == len(mat_mask)
            
            # get id list
            all_idx_list = [i.split('/')[-1].replace('.jpg','').split('_')[-1] for i in input_image ]
            base_name = []
            for idx, name in enumerate(input_image) :
                current_basename = os.path.basename(name)
                base_name.append(current_basename)
            gt_image = [os.path.join(gt_path,i).replace('_' + all_idx_list[idx] + '.jpg', '.jpg') for idx,i in enumerate(base_name)]
            if self.mat :
                mat_file = [os.path.join(gt_path,i).replace('_' + all_idx_list[idx] + '.jpg', '.jpg') for idx,i in enumerate(base_name)]
                assert len(input_image) == len(gt_image) == len(mat_file)
            else :
                assert len(input_image) == len(gt_image)
            
            for idx, each_image in enumerate(input_image) :
                
                # for input
                if class_idx in input_dict.keys():
                    input_dict[class_idx].append(input_image[idx])
                else:
                    input_dict[class_idx] = [input_image[idx]]
                
                # for gt
                if class_idx in gt_dict.keys():
                    gt_dict[class_idx].append(gt_image[idx])
                else:
                    gt_dict[class_idx] = [gt_image[idx]]
                    
                # for mat
                if self.mat :
                    if class_idx in gt_dict.keys():
                        mat_dict[class_id].append(mat_file[idx])
                    else:
                        mat_dict[class_id] = [mat_file[idx]]
        
        return input_dict, gt_dict, mat_dict

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.support_y_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        self.query_y_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            support_y = []
            query_x = []
            query_y = []
            # print(selected_cls)
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.input_data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                if not self.mat :
                    # print(np.array(self.input_data[cls])[indexDtrain].tolist())
                    support_x.append(
                        np.array(self.input_data[cls])[indexDtrain].tolist())  # get all input images filename for current Dtrain
                    query_x.append(
                        np.array(self.input_data[cls])[indexDtest].tolist())  # get all input images filename for current Dtest
                else :
                    support_x.append(
                        np.array(self.mat_data[cls])[indexDtrain].tolist())  # get all input mat filename for current Dtrain
                    query_x.append(
                        np.array(self.mat_data[cls])[indexDtest].tolist())  # get all input mat filename for current Dtest
                support_y.append(
                    np.array(self.gt_data[cls])[indexDtrain].tolist())  # get all gt images filename for current Dtrain
                query_y.append(
                    np.array(self.gt_data[cls])[indexDtest].tolist())  # get all gt images filename for current Dtest

            self.support_x_batch.append(support_x)  # append set to current sets
            self.support_y_batch.append(support_y)  # append sets to current sets
            self.query_x_batch.append(query_x)  # append set to current sets
            self.query_y_batch.append(query_y)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        if self.mode == "train" or self.mode == "validation" :
            # [setsz, 3, resize, resize]
            support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
            # [setsz]
            support_y = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
            # [querysz, 3, resize, resize]
            query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
            # [querysz]
            query_y = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)

            flatten_support_x = [sublist
                                 for sublist in self.support_x_batch[index] for item in sublist]
            flatten_support_y = [sublist
                                 for sublist in self.support_y_batch[index] for item in sublist]
            flatten_query_x = [sublist
                               for sublist in self.query_x_batch[index] for item in sublist]
            flatten_query_y = [sublist
                               for sublist in self.query_y_batch[index] for item in sublist]
            if self.mat :

                assert len(flatten_support_x) == len(flatten_support_y)
                for idx, path in enumerate(flatten_support_x):
                    current_mask = load_mat(path[0], 'light_mask')
                    support_y[i] = img_loader(flatten_support_y[idx], self.resize)
                    support_x[i] = generate_dark_image(support_y[i], current_mask)

                assert len(flatten_query_x) == len(flatten_query_y)
                for idx, path in enumerate(flatten_query_x):
                    current_mask = load_mat(path[0], 'light_mask')
                    query_y[i] = img_loader(flatten_query_y[idx], self.resize)
                    query_x[i] = generate_dark_image(support_y[i], current_mask)

            else :
                for i, path in enumerate(flatten_support_x):
                    support_x[i] = img_loader(path[0], self.resize)
                    current_example = support_x[i].permute(1,2,0).cpu().data.numpy()
                    current_example = current_example[...,::-1]*255
                    cv2.imwrite("feed_input.jpg",current_example.astype(np.uint8))
                for i, path in enumerate(flatten_support_y):
                    support_y[i] = img_loader(path[0], self.resize)
                    current_example = support_y[i].permute(1,2,0).cpu().data.numpy()
                    current_example = current_example[...,::-1]*255
                    cv2.imwrite("feed_gt.jpg",current_example.astype(np.uint8))
                for i, path in enumerate(flatten_query_x):
                    query_x[i] = img_loader(path[0], self.resize)

                for i, path in enumerate(flatten_query_y):
                    query_y[i] = img_loader(path[0], self.resize)
                
                    
            return support_x, support_y, query_x, query_y
        elif self.mode == "test" or self.mode == "train_normal" or self.mode == "val_normal" :
            
            x = torch.FloatTensor(3, self.resize, self.resize)
            y = torch.FloatTensor(3, self.resize, self.resize)
            x = img_loader(self.xname[index], self.resize)
            y = img_loader(self.yname[index], self.resize)
            name_input = os.path.basename(self.xname[index])
            name_truth = os.path.basename(self.yname[index])
            return x, y, name_input, name_truth
        else :
            raise ValueError("Wrong Data Directory !")

    def __len__(self):
        
        if self.mode == "train" or self.mode == "validation" :
            # as we have built up to batchsz of sets, you can sample some small batch size of sets.
            return self.batchsz
        elif self.mode == "train_normal" or self.mode == "val_normal":
            return len(self.xname)
        elif self.mode == "test" :
            return len(self.xname)
        else :
            raise ValueError("Wrong Data Directory !")

if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=5, k_shot=1, k_query=1, batchsz=1000, resize=168)

    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_

        support_x = make_grid(support_x, nrow=2)
        query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())
        plt.pause(0.5)

        tb.add_image('support_x', support_x)
        tb.add_image('query_x', query_x)

        time.sleep(5)

    tb.close()