# WAIT TO WRITE IN MULTIPORCESS VERSION
# =====================================

# import basic packages
import sys, cv2, glob, json, argparse, os, h5py, time
import numpy as np
from tqdm import tqdm

final_image, final_boxes, final_classes = [] ,[], []

# iter to retrieve data by batch size
def retrieve_data(file_names,batch_size,root_dir) :

    for index in range(len(file_names)):
        print("| Processing ==>\n[{}]\n===========".format(file_names[index]))
        with h5py.File(file_names[index], 'r') as hf:
            for each_dataset_name in hf :
                for dataset_detail in hf.get(each_dataset_name) :
                    total_data_length = hf[each_dataset_name][dataset_detail].shape[0]
                    print("| All Data Count : {}".format(total_data_length))
                    print("| Batch Size : {}".format(batch_size))

                    iter_time = (total_data_length//batch_size)
                    last_iter_patch = total_data_length%batch_size
                    
                    # deal with iter
                    for each_iter in range(iter_time) :
                        start = each_iter*batch_size
                        end = start+batch_size
                        print("\r| iter [ %d / %d ]" %(each_iter+1,iter_time+1),end="")
                        iter_data = hf[each_dataset_name][dataset_detail][start:end]
                        merge_data(iter_data,dataset_detail,root_dir)
                        sys.stdout.flush()
                    
                    # deal with last iter
                    print("\r| iter [ {} / {} ]".format(iter_time+1,iter_time+1))
                    iter_data = hf[each_dataset_name][dataset_detail][total_data_length-last_iter_patch:]
                    merge_data(iter_data,dataset_detail,root_dir)
                    sys.stdout.flush()

                    print("| Finishing -- {}\n============================="\
                        .format(dataset_detail))

# merge with exsit data
def merge_data(iter_data,dataset_detail,root_dir) :
    
    global final_image, final_boxes, final_classes

    with h5py.File(root_dir + "training_data.h5", "a") as hf:
        
        if list(hf.keys())==[] :
            training_data = hf.create_group('training_data')

            if dataset_detail == "images"  :
                iter_data = np.array(iter_data,dtype="uint8")
                training_data.create_dataset(dataset_detail,data=iter_data,compression="gzip",\
                            compression_opts=9,chunks=True,maxshape=(None,500,500,3))
            else :
                dt = h5py.special_dtype(vlen=str)
                iter_data = np.array(iter_data)
                training_data.create_dataset(dataset_detail,data=iter_data, dtype=dt,\
                            compression="gzip",\
                            compression_opts=9,chunks=True,maxshape=(None,))
        else :
            for data_group in hf :
                if dataset_detail in hf.get(data_group) :
                    if dataset_detail == "images" :
                        # print("1")
                        iter_data = np.array(iter_data,dtype="uint8")
                        hf[data_group][dataset_detail].resize(((hf[data_group][dataset_detail].shape[0])+len(iter_data)),axis=0)
                        hf[data_group][dataset_detail][-len(iter_data):] = iter_data
                    else :
                        dt = h5py.special_dtype(vlen=str)
                        iter_data = np.array(iter_data)
                        hf[data_group][dataset_detail].resize(((hf[data_group][dataset_detail].shape[0])+len(iter_data)),axis=0)
                        hf[data_group][dataset_detail][-len(iter_data):] = iter_data
                else :
                    if dataset_detail == "images"  :
                        # print("2")
                        iter_data = np.array(iter_data,dtype="uint8")
                        hf[data_group].create_dataset(dataset_detail,data=iter_data,compression="gzip",\
                                    compression_opts=9,chunks=True,maxshape=(None,500,500,3))
                    else :
                        dt = h5py.special_dtype(vlen=str)
                        iter_data = np.array(iter_data)
                        hf[data_group].create_dataset(dataset_detail,data=iter_data, dtype=dt,\
                                    compression="gzip",\
                                    compression_opts=9,chunks=True,maxshape=(None,))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_dir", default="/media/bill/BillData/比爾Bayesian-Satillite /dataset/Xview/dataset_500/",
                        help="Path to folder containing h5 training or validation data")
    parser.add_argument("--batch_size", default="3000", type=int,
                        help="Batch Size for Dealing with data every single time")
    
    args = parser.parse_args()
    root_dir = "/media/bill/BillData/比爾Bayesian-Satillite /dataset/Xview/"

    print("| Loading H5 file in directory ...\n===================")

    # load data
    file_names = glob.glob(args.h5_dir + "*.h5")
    file_names.sort()

    retrieve_data(file_names,args.batch_size,root_dir)