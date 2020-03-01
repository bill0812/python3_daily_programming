# TODO: Change HOME Path
import PIL, cv2, sys, torch
import numpy as np
import os.path as osp

from .config import HOME
from torch.utils.data.dataset import Dataset

# note: if you used our download scripts, this should be right
XVIEW_ROOT = osp.join(HOME, "data/")

def map_labels_contiguous(label_file):
	label_map = {}
	labels = open(label_file, 'r')
	for line in labels:
		ids = line.split(',')
		label_map[int(ids[0])] = int(ids[1])
	# print(label_map)
	return label_map


def map_labels_name(label_file):
	label_name = {}
	label_name_origin = {}
	labels = open(label_file, 'r')
	for line in labels:
		ids = line.split(',')
		# ids[2].split('\n')
		label_name[int(ids[1])] = ids[2].split('\n')[0]
		label_name_origin[int(ids[0])] = ids[2].split('\n')[0]
	# print(label_map)
	return label_name , label_name_origin


class XVIEWDetection(Dataset):
	"""XVIEW Detection Dataset Object

	input is image, target is annotation

	Arguments:
		root (string): filepath to XVIEW folder.
		image_set (string): imageset to use (eg. 'train', 'val', 'test')
		transform (callable, optional): transformation to perform on the
			input image
		target_transform (callable, optional): transformation to perform on the
			target `annotation`
			(eg: take in caption string, return tensor of word indices)
		dataset_name (string, optional): which dataset to load
			(default: 'XVIEW')
	"""

	def __init__(self, filename,
				 transform=None, target_transform=True,
				 dataset_name='XVIEW'):

		self.label_map = map_labels_contiguous(osp.join(XVIEW_ROOT, 'xview_labels.txt'))
		self.transform = transform
		self.target_transform = target_transform
		self.name = dataset_name

		self.result = {}
		
		# retrieve data from ground truth
		with open(filename,"r") as file_detail :
			for line, row in enumerate(file_detail) :
				
				# retrieve xmin, ymin, xmax, ymax, class_name
				img_name, xmin, ymin, xmax, ymax, class_name = row.split(",")
				xmin = int(float(xmin))
				ymin = int(float(ymin))
				xmax = int(float(xmax))
				ymax = int(float(ymax))
				class_name = int(float(class_name.split("\n")[0]))
			
				if (xmin, ymin, xmax, ymax, class_name) == ('', '', '', '', '') :
					continue

				if class_name == 18 :
					if img_name not in self.result :
						self.result[img_name] = []
					self.result[img_name].append(np.array([xmin, ymin, xmax, ymax, class_name]))

		self.images = list(self.result.keys())
		
	def __getitem__(self, i):

		# get each index data and ground truth
		img = cv2.imread(self.images[i])
		groundtruth = np.asarray(self.result[self.images[i]])
		
		self.boxes = groundtruth[:,0:4]
		self.classes = groundtruth[:,4:5]

		# how many bounding boxes, then you'll have that amount of classes
		assert self.boxes.shape[0] == self.classes.shape[0]
		
		height, width, channels = img.shape

		img_class = []
		if self.target_transform is True:
			# reshape the class
			img_class_xview = self.classes.reshape(-1)
			
			img_class = np.array([[self.label_map[int(x)]] if x in self.label_map else [self.label_map[0]] for x in img_class_xview])
			
			self.boxes = np.array(self.boxes).astype(np.float64)

			self.boxes[:, [0, 2]] /= width
			self.boxes[:, [1, 3]] /= height

		if self.transform is not None:
			bounding_box = np.array(self.boxes)
			img_class = np.array(img_class)
			img, boxes, labels = self.transform(img, bounding_box, img_class)

			# to rgb
			img = img[:, :, (2, 1, 0)]

			target = np.hstack((boxes, labels))
		# print(target)
		return torch.from_numpy(img).permute(2, 0, 1), target

	def __len__(self):
		return len(self.images)