import matplotlib.pyplot as plt 
import numpy as np
import glob
import os, sys, numpy

filepath_train_box = "../chipped_new_without_aug/boxes_500_train.npy"
filepath_train_classes = "../chipped_new_without_aug/classes_500_train.npy"
filepath_train_images = "../chipped_new_without_aug/images_500_train.npy"
filepath_val_box = "../chipped_new_without_aug/boxes_500_val.npy"
filepath_val_classes = "../chipped_new_without_aug/classes_500_val.npy"
filepath_val_images = "../chipped_new_without_aug/images_500_val.npy"

root = "Data/chipped/500px"
npyfilespath_boxes = [
                "boxes_500_train_1.npy",
                "boxes_500_train_2.npy"
                ] 

npyfilespath_classes = [
                "classes_500_train_1.npy",
                "classes_500_train_2.npy"
                ] 

npyfilespath_images = [
                "images_500_train_1.npy",
                "images_500_train_2.npy"
                ] 

# =======================================================================

npyfilespath_boxes_val = [
                "boxes_500_val_1.npy"
                ] 

npyfilespath_classes_val = [
                "classes_500_val_1.npy"
                ] 

npyfilespath_images_val = [
                "images_500_val_1.npy"
                ]   

all_data_boxes = []
all_data_classes = []
all_data_images = []
new_all_data_boxes = []
new_all_data_classes = []
new_all_data_images = []
get_rid_off_index = []
# for i in range(len(npyfilespath_boxes)):
#     for j in np.load(os.path.join(root, npyfilespath_boxes[i])) :
#         new_all_data_boxes.append(j)

# for i in range(len(npyfilespath_classes)):
#     for j in np.load(os.path.join(root, npyfilespath_classes[i])) :
#         new_all_data_classes.append(j)

# for i in range(len(npyfilespath_images)):
#     for j in np.load(os.path.join(root, npyfilespath_images[i])) :
#         new_all_data_images.append(j)

# print(len(all_data_boxes[0]))
# print(len(all_data_images))
# assert len(all_data_boxes) == len(all_data_classes) == len(all_data_images)

# for i in range(len(all_data_boxes)) :
#     # print(all_data_boxes[i])
#     # print(all_data_classes[i])
#     # print("==================================")
#     if all_data_boxes[i].sum() != 0 and all_data_classes[i].sum() != 0:

#         new_all_data_boxes.append(all_data_boxes[i])
#         new_all_data_classes.append(all_data_classes[i])
#         new_all_data_images.append(all_data_images[i])

# ===================================================================

all_data_boxes_val = []
all_data_classes_val = []
all_data_images_val = []
new_all_data_boxes_val = []
new_all_data_classes_val = []
new_all_data_images_val = []
get_rid_off_index_val = []
for i in range(len(npyfilespath_boxes_val)):
    for j in np.load(os.path.join(root, npyfilespath_boxes_val[i])) :
        new_all_data_boxes_val.append(j)

for i in range(len(npyfilespath_classes_val)):
    for j in np.load(os.path.join(root, npyfilespath_classes_val[i])) :
        new_all_data_classes_val.append(j)

for i in range(len(npyfilespath_images_val)):
    for j in np.load(os.path.join(root, npyfilespath_images_val[i])) :
        new_all_data_images_val.append(j)

# assert len(all_data_boxes_val) == len(all_data_classes_val) == len(all_data_images_val)

# for i in range(len(all_data_boxes_val)) :

#     if all_data_boxes_val[i].sum() != 0 and all_data_classes_val[i].sum() != 0:

#         new_all_data_boxes_val.append(all_data_boxes_val[i])
#         new_all_data_classes_val.append(all_data_classes_val[i])
#         new_all_data_images_val.append(all_data_images_val[i])

print(len(new_all_data_boxes))
print(len(new_all_data_classes))
print()
print(len(new_all_data_boxes_val))
print(len(new_all_data_classes_val))

# ============================================================

# print to check status
# print(all_data_boxes[0])
# print(len(all_data_boxes[0][0]))
# print(all_data_boxes)
# print(all_data_classes)
# print(len(all_data_images[0]))

# =============================================================

# np.save(filepath_train_box, new_all_data_boxes)
# np.save(filepath_train_classes, new_all_data_classes)
# np.save(filepath_train_images, new_all_data_images)
np.save(filepath_val_box,new_all_data_boxes_val)
np.save(filepath_val_classes, new_all_data_classes_val)
np.save(filepath_val_images, new_all_data_images_val)

# a = numpy.array([[  0.,   0.,  45.,  19.]])

# a /= 3

# print(a)