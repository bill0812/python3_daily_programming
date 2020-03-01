# Image Processing Tool 

### 👉 <font style="color:blue">   This is a command-line image processing tool about image processing with different filters.</font>

✨ <font style="color:orange">  ***Compare two method between " filter2d " function in opencv and "for-loop as normal (easy way)" method***</font>

### Some output by default : 

| Filter | Gray | RGB | HSV |
| :---: | :---: | :---: | :---: |
| `Sharpen` | ![Gray-sharpen](output-gray/output_1.jpg) | ![RGB-sharpen](output-rgb/output_1.jpg) | ![HSV-sharpen](output-hsv/output_1.jpg) |
| `Smooth`     | ![Gray-Smooth](output-gray/output_2.jpg) | ![RGB-Smooth](output-rgb/output_2.jpg) | ![HSV-Smooth](output-hsv/output_2.jpg) |
| `Outline`     | ![Gray-Outline](output-gray/output_3.jpg) | ![RGB-Outline](output-rgb/output_3.jpg) | ![HSV-Outline](output-hsv/output_3.jpg) |

---

### Inside, there are some files and directory :
    
>    1. [RGB.py](RGB.py) : Transfer to RGB first.

>    2. [HSV.py](HSV.py) : Transfer to HSV first.

>    3. [Gray.py](Gray.py) : Transfer to Gray Scale first. 

>    4. [main.py](main.py) : Main program to run.

>    5. [filter.json](filter.json) : Some Kernel you'd like to convolute with image.
>       *  Format would be like : 
>          ```json
>          "filter" :{
>               "1" : [[]],
>               "2" : [[]],
>               ...
>               "n" : [[]]
>          }
>            ```
>       *  The 2D-Array would be like 3 * 3 .

>  6.   [output](output/) : Default output directory after processing image.
>       *  Files include :
>          *  error.json : Calculate the distance between two method. (filter2d and for-loop as normal one), specificlly , I use MSE and RMSE to caculate error.
>          *  filterd images.

---

Simply Run below command :

```
$ python main.py --main.py --filter filter.json --image lena.jpg --choice 2
```

Also , there some helping mesage about parse arguments :

```
usage: Process Image With Some Filter Kernel. [-h] --filter
                                              path/to/filter.json --image
                                              path/to/image
                                              [--output path/to/output/directory]
                                              --choice {0,1,2}

optional arguments:
  -h, --help            show this help message and exit
  --filter path/to/filter.json
                        Enter your filter file to convolute the image.
  --image path/to/image
                        Input your image.
  --output path/to/output/directory
                        Output image to a specific directory
  --choice {0,1,2}      0 for gray image. 1 for RGB image. 2 for HSV image.
 ```