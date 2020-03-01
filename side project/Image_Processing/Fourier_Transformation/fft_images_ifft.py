import cv2, json, os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# part 2 : show only one phase or magnitude in single image
#============================================================
def operting_for_single_only_phase_spectrum(img_1,fshift_1,phase_1,out_dir,filename) :

    # creat a same size of original image using complex number
    # use the formula to get the real number and imaginary number, then combine them
    img_only_spectrum = np.multiply(np.abs(fshift_1), np.exp(1j*np.angle(0)))
    img_only_phase = np.multiply(np.abs(1), np.exp(1j*phase_1))

    # use ifft to transfer back from frequency domain to real number domain
    iffshift_only_spectrum = np.fft.ifftshift(img_only_spectrum)
    iffshift_only_spectrum_new = np.fft.ifft2(iffshift_only_spectrum)
    iffshift_only_phase = np.fft.ifftshift(img_only_phase)
    iffshift_only_phase_new = np.fft.ifft2(iffshift_only_phase)
    
    # due to the result is complex number, so turn to distanse, then normalize that
    # use real value
    iffshift_only_spectrum_new = np.real(iffshift_only_spectrum_new)

    # use absolute value
    # iffshift_only_magnitude_new = np.abs(iffshift_only_spectrum_new)
    min_1 = np.amin(iffshift_only_spectrum_new)
    range_1 = (np.amax(iffshift_only_spectrum_new) - np.amin(iffshift_only_spectrum_new))
    iffshift_only_spectrum_new = (iffshift_only_spectrum_new - min_1)/range_1

    # due to the result is complex number, so turn to distanse, then normalize that
    # use real value
    iffshift_only_phase_new = np.real(iffshift_only_phase_new)
    # use absolute value
    # iffshift_only_phase_new = np.abs(iffshift_only_phase_new)
    iffshift_only_phase_new = ((iffshift_only_phase_new-np.amin(iffshift_only_phase_new))/(np.amax(iffshift_only_phase_new)-np.amin(iffshift_only_phase_new)))

    # horizontally concat two image, then write
    iffshift_only_one = np.hstack((iffshift_only_spectrum_new,iffshift_only_phase_new))

    # show image and save
    cv2.imshow('ifft only spectrum (left) and phase (right)',iffshift_only_one)
    cv2.imwrite(out_dir + '/spectrum_and_phase_only_version_' + filename + '.jpg', iffshift_only_one*255)

    # wait for ESC key to exit
    if cv2.waitKey(0) == 27:         
        cv2.destroyAllWindows()

#===========================================================
# part 3 : combine two images using their phase and magnitude
def operting_for_two_swtich_phase_spectrum(img_1,fshift_1,fshift_2,phase_1,phase_2,out_dir) :

    # creat a same size of original image using complex number and
    # use the formula to get the real number and imaginary number, then combine them
    img_combine_1 = np.multiply(np.abs(fshift_1), np.exp(1j*phase_2))
    img_combine_2 = np.multiply(np.abs(fshift_2), np.exp(1j*phase_1))

    # use ifft to transfer back from frequency domain to real number domain
    img_combine_1 = np.fft.ifftshift(img_combine_1)
    img_combine_1_new = np.fft.ifft2(img_combine_1)
    img_combine_2 = np.fft.ifftshift(img_combine_2)
    img_combine_2_new = np.fft.ifft2(img_combine_2)

    # due to the result is complex number, so turn to distanse, then normalize that
    # use real value
    img_combine_1_new = np.real(img_combine_1_new)
    # use absolute value
    # img_combine_1_new = np.abs(img_combine_1_new)
    # normalize that
    img_combine_1_new = (img_combine_1_new-np.amin(img_combine_1_new))/(np.amax(img_combine_1_new)-np.amin(img_combine_1_new))
    
    # due to the result is complex number, so turn to distanse, then normalize that
    # use real value
    img_combine_2_new = np.real(img_combine_2_new)
    # use absolute value
    # img_combine_2_new = np.abs(img_combine_2_new)
    # normalize that
    img_combine_2_new = (img_combine_2_new-np.amin(img_combine_2_new))/(np.amax(img_combine_2_new)-np.amin(img_combine_2_new))

    # horizontally concat two image, then write
    img_combine = np.hstack((img_combine_1_new,img_combine_2_new))

    # show image
    cv2.imshow('imgage combine',img_combine)
    cv2.imwrite(out_dir + '/imgage_combine.jpg',img_combine*255)

    # wait for ESC key to exit
    if cv2.waitKey(0) == 27:         
        cv2.destroyAllWindows()

# ================================================
if __name__ == "__main__" :
    parser = ArgumentParser('Process Image With Fast Fourier Transformation : About Phase and Spectrum.')
    parser.add_argument('--firstimage', required = True , metavar='path/to/image', dest="img_name_1", help="Input your first image.")
    parser.add_argument('--secondimage', metavar='path/to/image', dest="img_name_2", help="Input your second image.")
    parser.add_argument('--output', metavar='path/to/output/directory', dest="out_dir", help="Output image to a specific directory", default="output_directory_default")
    parser.add_argument('--choice', required = True , type=int , choices = [0,1,2], dest="choice", help="0 for single show.\n1 for show single one aspect.\n2 for double switch.", default="2")
    args = parser.parse_args()

    # get filename
    img_name_1 = args.img_name_1
    filename = os.path.splitext(img_name_1)[0]
    img_name_2 = args.img_name_2

    # if second image 
    if img_name_2 == None :
        img_name_2 = img_name_1

    out_dir = args.out_dir
    choices = args.choice

    # create folder
    if not os.path.exists(out_dir) :
        os.makedirs(out_dir)

    # check if first image is a real image
    if ((img_name_1.endswith('.jpg')) or (img_name_1.endswith('.jpeg')) or (img_name_1.endswith('.png')) or (img_name_1.endswith('.tif'))):
        pass
    else:
        parser.error("file type is wrong !!")

    # get first image
    img_1 = cv2.imread(img_name_1,0)
    img_1 = cv2.resize(img_1,(512,512))
    f_1 = np.fft.fft2(img_1)
    fshift_1 = np.fft.fftshift(f_1)

    # use fft to get magnitude
    magnitude_spectrum_1 = np.log(np.abs(fshift_1))
    magnitude_spectrum_normalized_1 = (magnitude_spectrum_1-np.min(magnitude_spectrum_1))/(np.max(magnitude_spectrum_1)-np.min(magnitude_spectrum_1))

    # use fft to get phase
    phase_1 = np.angle(fshift_1)
    phase_normalized_1 = (phase_1-np.min(phase_1))/(np.max(phase_1)-np.min(phase_1))

    # if first image == second image
    if img_name_1 == img_name_2 :

        img_combine_original = img_1
        cv2.imwrite(out_dir + '/img_original_' + filename + '.jpg',img_combine_original)

    else :

        # get second image
        img_2 = cv2.imread(img_name_2,0)
        img_2 = cv2.resize(img_2,(512,512))
        f_2 = np.fft.fft2(img_2)
        fshift_2 = np.fft.fftshift(f_2)

        # use fft to get magnitude
        magnitude_spectrum_2 = np.log(np.abs(fshift_2))
        magnitude_spectrum_normalized_2 = (magnitude_spectrum_2-np.min(magnitude_spectrum_2))/(np.max(magnitude_spectrum_2)-np.min(magnitude_spectrum_2))

        # use fft to get phase
        phase_2 = np.angle(fshift_2)
        phase_normalized_2 = (phase_2-np.min(phase_2))/(np.max(phase_2)-np.min(phase_2))

        # horizontally concat two image, then write
        img_combine_original = np.hstack((img_1,img_2))

        cv2.imwrite(out_dir + '/img_combine_original_both.jpg',img_combine_original)

    # show image
    cv2.imshow('Imgage Original',img_combine_original)
    

    # part 1 : show phase and magnitude in single image
    if choices == 0 :

        # horizontally stack magnitude and phase
        magnitude_phase = np.hstack((magnitude_spectrum_normalized_1,phase_normalized_1))

        cv2.imshow('spectrum and phase',magnitude_phase)
        cv2.imwrite(out_dir + '/spectrum_and_phase_' + filename + '.jpg', magnitude_phase*255)

        # wait for ESC key to exit
        if cv2.waitKey(0) == 27:         
            cv2.destroyAllWindows()

    # ================================================
    # part 2 : show only one phase or magnitude in single image
    elif choices == 1 :
        
        operting_for_single_only_phase_spectrum(img_1,fshift_1,phase_1,out_dir,filename)

    # ================================================
    # part 3 : combine two images using their phase and magnitude
    elif choices == 2 :
        
        # if use option 2, you need different images
        if img_name_1 == img_name_2 :
            parser.error("\nYou need to Specifiy another image !!")
        
        else :
            
            operting_for_two_swtich_phase_spectrum(img_1,fshift_1,fshift_2,phase_1,phase_2,out_dir)      
    
    # ================================================
    else :
        parser.error("\nChoice is Wrong ( only 1 to 2 ) !!")