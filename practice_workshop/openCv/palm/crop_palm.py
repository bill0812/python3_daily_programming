import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def deal_img(image_name , output_dir):
    print("hello")
    # read image and transfer to scale / declare kernel
    img = cv.imread(image_name,cv.IMREAD_GRAYSCALE)
    kernel = np.ones((5,5),np.uint8)

    # use threshold by Gaussian or Mean
    # th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    #             cv.THRESH_BINARY,11,5)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,5)

    # use upper result to morphologyEx the opening
    opening = cv.morphologyEx(th3, cv.MORPH_OPEN, kernel)
    # closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    # dilated = cv.dilate(opening, kernel, iterations=2)

    # use the opening result to calculate the threshold again
    ret, mask = cv.threshold(opening, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)
    find_contours(mask , image_name , output_dir , img)

def find_contours(mask , image_name , output_dir , img) :

    # find Contours by mask
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # extract the " hand "
    for contour in contours:
        [x, y, w, h] = cv.boundingRect(contour)
        if w > 50 and h > 50 :
            crop_img = img[y:y+h, x:x+w]
            cv.imwrite(output_dir + image_name[8:-4] + "result.jpg",crop_img)


