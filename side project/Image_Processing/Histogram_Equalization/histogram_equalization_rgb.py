# importing opencv and  
import cv2 as cv
import numpy as np
import json, os
from math import sqrt
from matplotlib import pyplot as plt 

# define error
global error_dict
error_dict = dict()
error_dict["Error"] = dict()
  
def process_histogram_mine(img_name,out_dir):
    # reads an input image  and find frequency of pixels in range 0-255 
    img = cv.imread(img_name) 
    hist_b = cv.calcHist([img],[0],None,[256],[0,256]) 
    hist_g = cv.calcHist([img],[1],None,[256],[0,256]) 
    hist_r = cv.calcHist([img],[2],None,[256],[0,256]) 
    # plt.plot(hist_b) 
    # plt.plot(hist_g) 
    # plt.plot(hist_r) 
    # plt.show() 

    # define equalization map and L , and also get the height, width
    # also new the blank image
    equalization_map = dict()
    equalization_map["b"] = dict()
    equalization_map["g"] = dict()
    equalization_map["r"] = dict()
    equalization_val_list_b = list()
    equalization_val_list_g = list()
    equalization_val_list_r = list()
    L = 256-1
    height , width = img.shape[0] , img.shape[1]
    equalizaed_img = np.zeros((height,width,3), np.uint8)

    # process each channel
    print("Processing Histogram Equalization ...")
    print("=====================================")
    for i in range(3) :
        if i == 0 :
            equalizaed_img = process_each_channel(0,img,hist_b,equalization_map["b"],equalization_val_list_b,equalizaed_img,height,width,L,out_dir)
        elif i == 1 :
            equalizaed_img = process_each_channel(1,img,hist_g,equalization_map["g"],equalization_val_list_g,equalizaed_img,height,width,L,out_dir)
        elif i == 2 :
            equalizaed_img = process_each_channel(2,img,hist_r,equalization_map["r"],equalization_val_list_r,equalizaed_img,height,width,L,out_dir)
        else :
            pass
    print("Finishing Histogram Equalization ...")
    print("====================================")
    # send data to next progress
    print("Dealing OpenCv Tool , and show , then compare ...")
    print("=================================================")
    process_error_and_show(img,equalizaed_img,height,width,out_dir)

def process_each_channel(channel,img,hist,equalization_map,equalization_val_list,equalizaed_img,height,width,L,out_dir) :
    
    # start make the progress of histogram equalization
    status = 0
    while(status<=2) :
        for value in range(len(hist)) :

            # map the origin distribution after equalization
            if status == 0 :
                if value == 0 :
                    equalization_map[str(value)] = float(L*(hist[value]/(height*width)))
                else :
                    equalization_map[str(value)] = float(equalization_map[str(value-1)]) + float(L*(hist[value]/(height*width)))

                # change workflow
                if value == 255 :
                    status +=1

            # round the equalization map
            elif status == 1 :
                current_value = round(equalization_map[str(value)])
                equalization_map[str(value)] = current_value
                equalization_val_list.append(current_value)

                # change workflow
                if value == 255 :
                    status +=1

            # map the old image to histogram equalized image
            elif status == 2 :
                for x in range(height) :
                    for y in range(width) :
                        if img[x][y][channel] == value : 
                            equalizaed_img[x][y][channel] = equalization_val_list[value]
                
                # change workflow
                if value == 255 :
                    status +=1

    return  equalizaed_img
    
def process_error_and_show(img,equalizaed_img,height,width,out_dir) :
        
    # using opecv tool to equalizeHist
    opencv_equ = np.zeros((height,width,3), np.uint8)
    opencv_equ[:,:,0] = cv.equalizeHist(img[:,:,0])
    opencv_equ[:,:,1] = cv.equalizeHist(img[:,:,1])
    opencv_equ[:,:,2] = cv.equalizeHist(img[:,:,2])

    # plot the all historgram for each image
    hist_v_original_r = cv.calcHist([img],[0],None,[256],[0,256]) 
    hist_v_original_g = cv.calcHist([img],[1],None,[256],[0,256]) 
    hist_v_original_b = cv.calcHist([img],[2],None,[256],[0,256]) 
    hist_v_opencvtool_r = cv.calcHist([opencv_equ],[0],None,[256],[0,256]) 
    hist_v_opencvtool_g = cv.calcHist([opencv_equ],[1],None,[256],[0,256]) 
    hist_v_opencvtool_b = cv.calcHist([opencv_equ],[2],None,[256],[0,256]) 
    hist_v_mine_r = cv.calcHist([equalizaed_img],[0],None,[256],[0,256]) 
    hist_v_mine_g = cv.calcHist([equalizaed_img],[1],None,[256],[0,256]) 
    hist_v_mine_b = cv.calcHist([equalizaed_img],[2],None,[256],[0,256]) 

    # plot r histogram
    plt.subplot(1, 3, 1)
    plt.plot(hist_v_original_r,color="r",linestyle=':',label='original image') 
    plt.plot(hist_v_opencvtool_r,color="g",linestyle='--',label='opencv tool') 
    plt.plot(hist_v_mine_r,color="b",label='my method')
    plt.title("Plot R histogram")
    plt.legend(loc='lower left')

    # plot g histogram
    plt.subplot(1, 3, 2)
    plt.plot(hist_v_original_g,color="r",linestyle=':',label='original image') 
    plt.plot(hist_v_opencvtool_g,color="g",linestyle='--',label='opencv tool') 
    plt.plot(hist_v_mine_g,color="b",label='my method') 
    plt.title("Plot G histogram")
    plt.legend(loc='lower left')

    # plot b histogram
    plt.subplot(1, 3, 3)
    plt.plot(hist_v_original_b,color="r",linestyle=':',label='original image') 
    plt.plot(hist_v_opencvtool_b,color="g",linestyle='--',label='opencv tool') 
    plt.plot(hist_v_mine_b,color="b",label='my method') 
    plt.title("Plot B histogram")
    plt.legend(loc='lower left')
    plt.savefig( (out_dir + "/"  + "histogram.jpg") )
    plt.show()

    # show all image that me make above
    image_all = np.hstack((img,opencv_equ,equalizaed_img))
    cv.imshow("show all img (left : original ; center : opencv tool ; right : mine)",image_all)

    # calculate the error between opencv tool and mine
    # make two kinds image to (0,1), and define some variable
    equalizaed_img = equalizaed_img/255
    opencv_equ = opencv_equ/255
    current_error_mse = 0
    current_error_rmse = 0

    # caculate error with mse\
    for i in range(3) :
        err = np.sum((equalizaed_img[:,:,i].astype("float") - opencv_equ[:,:,i].astype("float")) ** 2)
        err = err / float(height * width)
        current_error_mse = current_error_mse + err
        current_error_rmse = sqrt(current_error_mse)

    print("error : {\n\tMSE : %.10f,\n\tRMSE : %.10f,\n}" %(current_error_mse,current_error_rmse))
    
    # save result
    error_dict["Error"]["MSE"] = current_error_mse
    error_dict["Error"]["RMSE"] = current_error_rmse

    # write image for all
    cv.imwrite( (out_dir + "/"  + "compared_histogram_each.jpg") , image_all )

    # write error file
    with open(out_dir+ '/error.json', 'w') as outfile:
        json.dump(error_dict, outfile)
    if cv.waitKey(0) & 0xFF == ord('q'):
        cv.destroyAllWindows()
