import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import *
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
from math import sqrt
import cv2 , json

global error_dict
error_dict = dict()
error_dict["Error"] = dict()
error_dict["Error"]["MSE"] = dict()
error_dict["Error"]["RMSE"] = dict()

def open_image_and_filter(filter_name,img_name,out_dir) :

    img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Lena Original', img)
    img_padded = np.pad(img, ((1,1), (1,1)), mode='constant')
    new_img = np.zeros(img_padded.shape)
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
                new_img[j+1,k+1] = np.dot(img_padded[j:j+3,k:k+3].flatten() , filter_matrix.flatten())
                if new_img[j+1,k+1] <= 0 :
                    new_img[j+1,k+1] = 0
                elif new_img[j+1,k+1] >= 255 :
                    new_img[j+1,k+1] = 255
                else :
                    pass

    filtered_img_show(new_img,out_dir,count,filter_matrix,img)

def filtered_img_show(new_img,out_dir,count,filter_matrix,img):
    filter2D_img = cv2.filter2D(img,-1,filter_matrix)
    new_img = new_img/255  
    current_error_mse = 0
    current_error_rmse = 0

    cv2.imshow('Lena_Processed_filter2D_' + str(count), filter2D_img)
    cv2.imshow('Lena_Processed_' + str(count), new_img)

    # caculate error with mse
    filter2D_img = filter2D_img / filter2D_img.max()
    current_error_mse = np.sum((filter2D_img.astype("float") - new_img[1:-1,1:-1].astype("float")) ** 2)
    current_error_mse = current_error_mse / float(filter2D_img.shape[0] * filter2D_img.shape[1])
    current_error_rmse = sqrt(current_error_mse)

    print(" ----- ----------- ----- \n error : {\n\tMSE : %.10f,\n\tRMSE : %.10f,\n}" %(current_error_mse,current_error_rmse))
    
    error_dict["Error"]["MSE"][str(count+1)] = current_error_mse
    error_dict["Error"]["RMSE"][str(count+1)] = current_error_rmse

    cv2.imwrite( (out_dir + "/" + out_dir + "_" + str(count+1) + ".jpg") , new_img*255 )
