from PIL import Image
from numpy import array
img = Image.open("/Users/wangboren/Desktop/Screen Shot 2018-06-19 at 5.14.55 PM.png").resize((28,28), Image.BILINEAR)
arr = array(img)
print(arr[0])
print(arr.reshape(1, 56, 56, 1)[0][0].shape)

import numpy as np
