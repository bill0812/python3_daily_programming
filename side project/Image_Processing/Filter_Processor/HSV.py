import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import *
from math import sqrt
import cv2 , json

global scaler
scaler = MinMaxScaler(feature_range=(0, 255))

global error_dict
error_dict = dict()
error_dict["Error"] = dict()
error_dict["Error"]["MSE"] = dict()
error_dict["Error"]["RMSE"] = dict()

def open_image_and_filter(filter_name,img_name,out_dir) :
    
    img = cv2.imread(img_name)
    img_new = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    cv2.imshow('Lena Original', img_new)
    img_padded = np.pad(img_new, ((1,1), (1,1), (0,0)), mode='constant')
    new_img = np.zeros(img_padded.shape)
    new_img[:,:,0:2] = img_padded[:,:,0:2]
    with open(filter_name) as f:
        data = json.load(f)

    for i in range(len(data["filter"])) :
        image_filter(np.array(data["filter"][str(i+1)]),img_padded,new_img,out_dir,i,img)

    with open(out_dir+ '/error.json', 'w') as outfile:
        json.dump(error_dict, outfile)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def image_filter(filter_matrix,img_padded,new_img,out_dir,count,img):

    divide = np.sum(filter_matrix)

    if divide > 0 or divide < 0 :
        filter_matrix = filter_matrix / divide
    
    print("----- filter is : -----\n",filter_matrix)

    for j in range(512) :
        for k in range(512) :
                new_img[j+1,k+1,2] = np.dot(img_padded[j:j+3,k:k+3,2].flatten() , filter_matrix.flatten())
                if new_img[j+1,k+1,2] <= 0 :
                    new_img[j+1,k+1,2] = 0
                elif new_img[j+1,k+1,2] >= 255 :
                    new_img[j+1,k+1,2] = 255
                else :
                    pass

    flat_temp = new_img[1:-1,1:-1,2].flatten().reshape(-1,1)
    scaler.fit(flat_temp)
    flat_temp = scaler.transform(flat_temp)
    new_img[1:-1,1:-1,2] = flat_temp.flatten().reshape([512,512])
    filtered_img_show(new_img,out_dir,count,filter_matrix,img)

def filtered_img_show(new_img,out_dir,count,filter_matrix,img):

    filter2D_img = cv2.filter2D(img,-1,filter_matrix)
    current_error_mse = 0
    current_error_rmse = 0

    fine_tune_img = cv2.cvtColor(new_img.astype(np.uint8), cv2.COLOR_HSV2RGB)    

    cv2.imshow('Lena_Processed_filter2D_' + str(count), filter2D_img)
    cv2.imshow('Lena_Processed_' + str(count+1), fine_tune_img)

    # caculate error with mse
    for i in range(3) :
        filter2D_img[:,:,i] = filter2D_img[:,:,i] / filter2D_img[:,:,i].max()
        fine_tune_img[:,:,i] = fine_tune_img[:,:,i] / fine_tune_img[:,:,i].max()
        err = np.sum((filter2D_img[:,:,i].astype("float") - fine_tune_img[1:-1,1:-1,i].astype("float")) ** 2)
        err = err / float(filter2D_img.shape[0] * filter2D_img.shape[1])
        current_error_mse = current_error_mse + err
        current_error_rmse = sqrt(current_error_mse)
    
    print(" ----- ----------- ----- \n error : {\n\tMSE : %.10f,\n\tRMSE : %.10f,\n}" %(current_error_mse,current_error_rmse))
    
    error_dict["Error"]["MSE"][str(count+1)] = current_error_mse
    error_dict["Error"]["RMSE"][str(count+1)] = current_error_rmse

    cv2.imwrite(out_dir + "/" + out_dir + "_" + str(count+1) + ".jpg", fine_tune_img)
