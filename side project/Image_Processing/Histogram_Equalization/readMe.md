# Image Histogram Equalization Tool 

### <font style="color:blue">This is a command-line image processing tool about Histogram Equalization.</font>

### Some output by default : 

| Style | Image Show <br> ( left : original ; center : opencv tool ; right : mine ) | Image Histogram |
| :---: | :---: | :---: | 
| `Gray` | ![Gray-Image](out_dir_gray/compared_histogram_each.jpg) | ![Gray-Hist](out_dir_gray/histogram.jpg) |
| `RGB`  | ![RGB-Image](out_dir_rgb/compared_histogram_each.jpg) | ![RGB-Hist](out_dir_rgb/histogram.jpg) |
| `HSV`  | ![HSV-Image](out_dir_hsv/compared_histogram_each.jpg) | ![HSV-Hist](out_dir_hsv/histogram.jpg) |
| `YCbCr` | ![YCbCr-Image](out_dir_ycbcr/compared_histogram_each.jpg) | ![YCbCr-Hist](out_dir_ycbcr/histogram.jpg) |

---

### Inside, there are some files and directory :
    
>    1. [histogram_equalization_grayscale.py](histogram_equalization_grayscale.py) : Transfer to _GRAY_ first.

>    2. [histogram_equalization_rgb.py](histogram_equalization_rgb.py) : Transfer to _RGB_ first.

>    3. [histogram_equalization_hsv.py](histogram_equalization_hsv.py) : Transfer to _HSV_ Scale first. 

>    4. [histogram_equalization_ycbcr.py](histogram_equalization_ycbcr.py) : Transfer to _YCbCr_ Scale first. 

>    5. [main.py](main.py) : Main program to run.

>    6. [out_dir_gray](out_dir_gray/) : Default output directory after processing _"GRAY"_ image.

>    7. [out_dir_rgb](out_dir_rgb/) : Default output directory after processing _"RGB"_ image.

>    8. [out_dir_hsv](out_dir_hsv/) : Default output directory after processing _"HSV"_ image.

>    9. [out_dir_ycbcr](out_dir_ycbcr/) : Default output directory after processing _"YCbCr"_ image.

>    10. [error.json](out_dir_ycbcr/error.json) : Take directory "out_dir_ycbcr" for example, we caculate the MSE and RMSE between two images, which is from "OpenCv Tool" and "Mine"

>   11. [compared_histogram_each.jpg](out_dir_ycbcr/compared_histogram_each.jpg) /  [histogram.jpg](out_dir_ycbcr/histogram.jpg): Take directory "out_dir_ycbcr" for example, output two images after the caculation. They are each images after histogram equalized and each images' histogram comparsion.

---

Simply Run below command :

```
$ python main.py --main.py --image mp2.jpg --choice 0
```
or

```
$ python main.py --main.py --image mp2a.jpg --choice 1(or 2 , 3)
```

Also , there some helping mesage about parse arguments :

```
usage: Process Image With Histogram Equalization. 

[-h] --image path/to/image
[--output path/to/output/directory]
--choice {0,1,2,3}
 ```

##  Here is a simply introduction of this tool.
