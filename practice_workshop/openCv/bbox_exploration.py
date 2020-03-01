import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


#height = []
#width = []
#with open('train.csv', newline='') as csvfile:

#    rows = csv.reader(csvfile)

#    for idx, row in enumerate(rows):
#        if idx == 0 : break
#            x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
#            width.append(x2-x1)
#            height.append(y2-y1)

df_train = pd.read_csv('train.csv', header=None, names=['path', 'x1', 'y1', 'x2', 'y2', 'object_name'])

print(df_train.columns)
'''
## width
print('>> width statistics:')
width = df_train['x2']-df_train['x1']
print(width.describe())
width_sns_plot = sns.distplot(width, kde=False, rug=True)
width_sns_plot.figure.savefig("width_stats.png")
#skewness and kurtosis
print("Skewness: %f" % width.skew())
print("Kurtosis: %f" % width.kurt())
plt.close()

## height
print('>> height statistics:')

height = df_train['y2']-df_train['y1']
print(height.describe())
height_sns_plot = sns.distplot(height, kde=False, rug=True)
height_sns_plot.figure.savefig("height_stats.png")
#skewness and kurtosis
print("Skewness: %f" % height.skew())
print("Kurtosis: %f" % height.kurt())
plt.close()

## area
print('>> area statistics:')
area = width * height
print(area.describe())
area_sns_plot = sns.distplot(area, kde=False, rug=True)
area_sns_plot.figure.savefig("area_stats.png")
#skewness and kurtosis
print("Skewness: %f" % area.skew())
print("Kurtosis: %f" % area.kurt())
plt.close()

'''

'''
## obj number in all image
print(df_train['object_name'].value_counts())

## obj number in per image
print('>> obj number in per image:')
obj_per_image = df_train['path'].value_counts()
print(obj_per_image.describe())
obj_per_image_sns_plot = sns.distplot(obj_per_image, rug=True)
obj_per_image_sns_plot.figure.savefig("obj_per_image_stats.png")
#skewness and kurtosis
print("Skewness: %f" % obj_per_image.skew())
print("Kurtosis: %f" % obj_per_image.kurt())
plt.close()

## total image number with gt
print('>> total image number with gt')
print(np.shape(pd.unique(df_train['path'])))
'''


