import RGB , HSV , Gray
from argparse import ArgumentParser
import json, os

if __name__ == "__main__" :
    parser = ArgumentParser('Process Image With Some Filter Kernel.')
    parser.add_argument('--filter', required = True , metavar='path/to/filter.json', dest="filter_name", help="Enter your filter file to convolute the image.")
    parser.add_argument('--image', required = True , metavar='path/to/image', dest="img_name", help="Input your image.")
    parser.add_argument('--output', metavar='path/to/output/directory', dest="out_dir", help="Output image to a specific directory", default="output")
    parser.add_argument('--choice', required = True , type=int , choices = [0,1,2], dest="choice", help="0 for gray image.\n1 for RGB image.\n2 for HSV image.", default="0")
    args = parser.parse_args()
    filter_name = args.filter_name
    img_name = args.img_name
    out_dir = args.out_dir
    choices = args.choice

    if (filter_name.endswith('.json')) and ((img_name.endswith('.jpg')) or (img_name.endswith('.jpeg')) or (img_name.endswith('.png')) ):
        pass
    else:
        parser.error("file type for image or filter is wrong !!")

    if not os.path.exists(out_dir) :
        os.makedirs(out_dir)

    if choices == 0 :
        Gray.open_image_and_filter(filter_name,img_name,out_dir)
    elif choices == 1 :
        RGB.open_image_and_filter(filter_name,img_name,out_dir)
    elif choices == 2 :
        HSV.open_image_and_filter(filter_name,img_name,out_dir)
    else :
        parser.error("Choice is Wrong ( only 1 to 3 ) !!")
