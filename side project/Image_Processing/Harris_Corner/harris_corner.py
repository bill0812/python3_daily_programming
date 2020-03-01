import cv2, json, os
import numpy as np

def gradient_x(image_gray,height,width):
    ##Sobel operator kernels.
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    image_gray_padded = np.pad(image_gray, ((1,1), (1,1)), mode='constant')
    new_image_gray = np.zeros(image_gray_padded.shape)
    for j in range(height) :
        for k in range(width) :
            new_image_gray[j+1,k+1] = np.dot(image_gray_padded[j:j+3,k:k+3].flatten() , kernel_x.flatten())
            if new_image_gray[j+1,k+1] <= 0 :
                new_image_gray[j+1,k+1] = 0
            elif new_image_gray[j+1,k+1] >= 255 :
                new_image_gray[j+1,k+1] = 255
            else :
                pass
    # show image and save
    cv2.imshow("gradient x", new_image_gray/255)
    return new_image_gray

def gradient_y(image_gray,height,width):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    image_gray_padded = np.pad(image_gray, ((1,1), (1,1)), mode='constant')
    new_image_gray = np.zeros(image_gray_padded.shape)
    for j in range(height) :
        for k in range(width) :
            new_image_gray[j+1,k+1] = np.dot(image_gray_padded[j:j+3,k:k+3].flatten() , kernel_y.flatten())
            if new_image_gray[j+1,k+1] <= 0 :
                new_image_gray[j+1,k+1] = 0
            elif new_image_gray[j+1,k+1] >= 255 :
                new_image_gray[j+1,k+1] = 255
            else :
                pass

    # show image and save
    cv2.imshow("gradient y", new_image_gray/255)

    return new_image_gray

if __name__ == "__main__":
    
    # read image as grayscale and also RGB style
    # and read the height and width
    # image_rgb_mine = cv2.imread("corner.jpg",1)
    # image_rgb_cv = image_rgb_mine
    # image_gray = cv2.imread("corner.jpg",0)
    # image_rgb_mine = cv2.imread("lab.jpg",1)
    # image_rgb_cv = image_rgb_mine
    # image_gray = cv2.imread("lab.jpg",0)
    image_rgb_mine = cv2.imread("test.png",1)
    image_rgb_cv = image_rgb_mine
    image_gray = cv2.imread("test.png",0)
    height = image_gray.shape[0]
    width = image_gray.shape[1]

    # show original one
    cv2.imshow('Original One',image_rgb_mine)

    print("Processing... !!!")

    # use the cv function to compare
    image_gray = np.float32(image_gray)
    corner_harris_cv = cv2.cornerHarris(image_gray,2,3,0.04)
    # corner_harris_cv = cv2.dilate(corner_harris_cv, None, iterations=3)

    # we use sobel operators to get the x-align gradient and y-align gradient
    # using covolution function we use in HW1
    I_x = gradient_x(image_gray,height,width)
    I_y = gradient_y(image_gray,height,width)

    # as the definition, that is the formula
    # we let Ixx, Ixy and Iyy as below
    Ixx = I_x**2
    Ixy = I_y*I_x
    Iyy = I_y**2

    # set some variables, 
    # like window size that compute between Ixx and Iyy and Ixy
    # and offset that we move(slide) how much
    window_size = 3
    offset = int(window_size/2)

    # make the new r numpy array for Harris operation (response)
    new_r_image = np.zeros(image_gray.shape)

    # compute the Sxx, Syy, Sxy as the formula mentioned
    # using slide window to cmpute 
    for x in range(offset, height-offset):
        for y in range(offset, width-offset):

            # Ixx[ width ; height ] ; Ixy[ width ; height ] ; Iyy[ width ; height ]
            # 內積
            Sxx = np.sum(Ixx[x-offset:x+1+offset, y-offset:y+1+offset])
            Syy = np.sum(Iyy[x-offset:x+1+offset, y-offset:y+1+offset])
            Sxy = np.sum(Ixy[x-offset:x+1+offset, y-offset:y+1+offset])

            # Find determinant and trace, use to get corner (response)
            # k is the parameters for the formula
            k = 0.04
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy

            # the most important is here, compute the "r"
            r = det - k*(trace**2)

            # reset the new_r_image list as r
            new_r_image[x][y] = r
            # set for my function
            # if r > 10000 :
            #     cv2.circle(image_rgb_mine, (y, x), 1, (0, 0, 255), -1)


    # set a threshold and compare to current r value
    threshold = 0.005
    for i in range(height) :
        for j in range(width) :

            # circle function for center is (width_place, height_place)
            # so we have to inver (i,j) to (j,i)

            # set for my function
            if new_r_image[i][j] > threshold*new_r_image.max() :
                cv2.circle(image_rgb_mine, (j, i), 1, (0, 0, 255), -1)

            # set for cv function
            if corner_harris_cv[i][j] > threshold*corner_harris_cv.max() :
                cv2.circle(image_rgb_cv, (j, i), 1, (0, 0, 255), -1)
    
    combine_all = np.hstack((image_rgb_mine,image_rgb_cv))

    # show image and save
    # cv2.imshow('compare two ways (lab)',combine_all)
    # cv2.imwrite('combine_all_lab.jpg',combine_all)

    cv2.imshow('compare two ways (test)',combine_all)
    cv2.imwrite('combine_all_test.png',combine_all)

    # cv2.imshow('compare two ways (blackandwhite)',combine_all)
    # cv2.imwrite('combine_all_blackandwhite.jpg',combine_all)
   
    print("Finish !!!")

    # wait for ESC key to exit
    if cv2.waitKey(0) == 27:         
        cv2.destroyAllWindows()