import os
import numpy

from skimage import io
from skimage import color

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CamLocDataset(Dataset):
	"""Camera localization dataset.

	Access to image, calibration and ground truth data given a dataset directory.
	Training flag indicates wether initialization targets should be loaded.
	"""

	def __init__(self, root_dir, training=True):

		self.training = training

		rgb_dir = root_dir + '/rgb/'
		pose_dir =  root_dir + '/poses/'
		calibration_dir = root_dir + '/calibration/'
		coord_dir =  root_dir + '/init/'

		self.rgb_files = os.listdir(rgb_dir)
		self.rgb_files = [rgb_dir + f for f in self.rgb_files]
		self.rgb_files.sort()

		self.image_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.4, 0.4, 0.4], # statistics calculated over 7scenes training set
				std=[0.25, 0.25, 0.25]
				)
			])

		self.pose_files = os.listdir(pose_dir)
		self.pose_files = [pose_dir + f for f in self.pose_files]
		self.pose_files.sort()

		self.pose_transform = transforms.Compose([
			transforms.ToTensor()
			])

		self.calibration_files = os.listdir(calibration_dir)
		self.calibration_files = [calibration_dir + f for f in self.calibration_files]
		self.calibration_files.sort()		

		if self.training:
			self.coord_files = os.listdir(coord_dir)
			self.coord_files = [coord_dir + f for f in self.coord_files]
			self.coord_files.sort()

		if len(self.rgb_files) != len(self.pose_files):
			raise Exception('RGB file count does not match pose file count!')

	def __len__(self):
		return len(self.rgb_files)

	def __getitem__(self, idx):

		image = io.imread(self.rgb_files[idx])

		if len(image.shape) < 3:
			image = color.gray2rgb(image)
		image = self.image_transform(image)	

		focal_length = float(numpy.loadtxt(self.calibration_files[idx]))

		pose = numpy.loadtxt(self.pose_files[idx])
		pose = torch.from_numpy(pose).float().inverse()

		if self.training:
			coords = torch.load(self.coord_files[idx])
		else:
			coords = 0

		return image, pose, coords, focal_length, self.rgb_files[idx]
