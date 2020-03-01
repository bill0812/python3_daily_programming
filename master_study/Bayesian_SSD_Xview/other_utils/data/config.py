# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("/home/bill/Desktop/research_bayesian_bigscale/Bayesian_SSD_Xview/SSD/other_utils")
# HOME = os.path.expanduser("/home/bill/Desktop/research_bayesian_bigscale/Bayesian_SSD_Xview/SSD")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
		  (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
xview = {
	'num_classes': 61,
	'lr_steps': (300, 600, 900, 1200, 1500, 1800),
	'max_iter': 1800,
	'feature_maps': [38, 19, 10, 5, 3, 1],
	'min_dim': 300,
	'steps': [8, 16, 32, 64, 100, 300],
	'min_sizes': [21, 45, 99, 153, 207, 261],
	'max_sizes': [45, 99, 153, 207, 261, 315],
	'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
	'variance': [0.1, 0.2],
	'clip': True,
	'name': 'XVIEW',
}
