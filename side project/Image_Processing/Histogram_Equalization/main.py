import histogram_equalization_grayscale as gray
import histogram_equalization_rgb as rgb
import histogram_equalization_hsv as hsv
import histogram_equalization_ycbcr as ycbcr
from argparse import ArgumentParser
import json, os

if __name__ == "__main__" :
    parser = ArgumentParser('Process Image With Histogram Equalization.')
    parser.add_argument('--image', required = True , metavar='path/to/image', dest="img_name", help="Input your image.")
    parser.add_argument('--output', metavar='path/to/output/directory', dest="out_dir", help="Output image to a specific directory", default="output")
    parser.add_argument('--choice', required = True , type=int , choices = [0,1,2,3], dest="choice", help="0 for gray image.\n1 for RGB image.\n2 for HSV image.", default="0")
    args = parser.parse_args()
    img_name = args.img_name
    out_dir = args.out_dir
    choices = args.choice

    if ((img_name.endswith('.jpg')) or (img_name.endswith('.jpeg')) or (img_name.endswith('.png')) ):
        pass
    else:
        parser.error("file type for image or filter is wrong !!")

    if choices == 0 :
        out_dir = "out_dir_gray"
        if not os.path.exists(out_dir) :
            os.makedirs(out_dir)
        gray.process_histogram_mine(img_name,out_dir)
    elif choices == 1 :
        out_dir = "out_dir_rgb"
        if not os.path.exists(out_dir) :
            os.makedirs(out_dir)
        rgb.process_histogram_mine(img_name,out_dir)
    elif choices == 2 :
        out_dir = "out_dir_hsv"
        if not os.path.exists(out_dir) :
            os.makedirs(out_dir)
        hsv.process_histogram_mine(img_name,out_dir)        
    elif choices == 3 :
        out_dir = "out_dir_ycbcr"
        if not os.path.exists(out_dir) :
            os.makedirs(out_dir)
        ycbcr.process_histogram_mine(img_name,out_dir)        
    else :
        parser.error("Choice is Wrong ( only 1 to 3 ) !!")
