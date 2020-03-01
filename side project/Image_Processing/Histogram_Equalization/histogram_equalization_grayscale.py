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
    img = cv.imread(img_name,cv.IMREAD_GRAYSCALE) 
    hist = cv.calcHist([img],[0],None,[256],[0,256]) 

    # define equalization map and L , and also get the height, width
    # also new the blank image
    equalization_map = dict()
    equalization_val_list = list()
    L = 256-1
    height , width = img.shape[0] , img.shape[1]
    equalizaed_img = np.zeros((height,width), np.uint8)

    # start make the progress of histogram equalization
    status = 0
    print("Processing Histogram Equalization ...")
    print("=====================================")
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
                        if img[x][y] == value : 
                            equalizaed_img[x][y] = equalization_val_list[value]
                
                # change workflow
                if value == 255 :
                    status +=1

    print("Finishing Histogram Equalization ...")
    print("====================================")

    # send data to next progress
    print("Dealing OpenCv Tool , and show , then compare ...")
    print("=================================================")
    process_error_and_show(img,equalizaed_img,height,width,out_dir)
    
def process_error_and_show(img,equalizaed_img,height,width,out_dir) :
        
    # using opecv tool to equalizeHist
    opencv_equ = cv.equalizeHist(img)

    # plot the all historgram for each image
    hist_v_original = cv.calcHist([img],[0],None,[256],[0,256]) 
    hist_v_opencvtool = cv.calcHist([opencv_equ],[0],None,[256],[0,256]) 
    hist_v_mine = cv.calcHist([equalizaed_img],[0],None,[256],[0,256]) 
    plt.plot(hist_v_original,color="r",linestyle=':',label='original image') 
    plt.plot(hist_v_opencvtool,color="g",linestyle='--',label='opencv tool') 
    plt.plot(hist_v_mine,color="b",label='my method')
    plt.title("Plot histogram")
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

    # caculate error with mse
    current_error_mse = np.sum((equalizaed_img.astype("float") - opencv_equ.astype("float")) ** 2)
    current_error_mse = current_error_mse / float(height * width)
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
