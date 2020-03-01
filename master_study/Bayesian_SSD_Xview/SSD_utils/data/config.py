# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("/home/bill/Desktop/research_bayesian_bigscale/Bayesian_SSD_Xview/SSD/SSD_utils")
# HOME = os.path.expanduser("/home/bill/Desktop/research_bayesian_bigscale/Bayesian_SSD_Xview/SSD")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
		  (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
xview = {
	'num_classes': 2,
	'lr_steps': (400, 800, 1200),
	'max_iter': 1200,
	'feature_maps': [38, 19, 10, 5, 3, 1],
	'min_dim': 300,
	'steps': [8, 16, 32, 64, 100, 300],
	'min_sizes': [21, 45, 99, 153, 207, 261],
	'max_sizes': [45, 99, 153, 207, 261, 315],
	'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],#[[1., 2., 0.5], [1., 2., 3., 0.5, .333], [1., 2., 3., 0.5, .333], [1., 2., 3., 0.5, .333], [1., 2., 0.5], [1., 2., 0.5]],
	'variance': [0.1, 0.2],
	'clip': True,
	'name': 'XVIEW',
}
