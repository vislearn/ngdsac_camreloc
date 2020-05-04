import torch
import torch.optim as optim

import time
import argparse
import math

from dataset import CamLocDataset
from network import Network
import util

parser = argparse.ArgumentParser(
	description='Train scene coordinate regression by minimizing re-projection error.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder')

parser.add_argument('network_in', help='file name of a network initialized for the scene')

parser.add_argument('network_out', help='output file name for the new network')

parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, 
	help='learning rate')

parser.add_argument('--iterations', '-iter', type=int, default=300000,
	help='number of training iterations, i.e. numer of model updates')

parser.add_argument('--softclamp', '-sc', type=float, default=10, 
	help='robust square root loss after this threshold, in px')

parser.add_argument('--hardclamp', '-hc', type=float, default=100, 
	help='clamp loss with this threshold, in px')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files, useful to separate different runs of a script')

opt = parser.parse_args()

trainset = CamLocDataset("./dataset/" + opt.scene + "/train")
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6)

print("Found %d training images for %s." % (len(trainset), opt.scene))

# load network
network = Network(torch.zeros((3)))
network.load_state_dict(torch.load(opt.network_in))
network = network.cuda()
network.train()

print("Successfully loaded %s." % opt.network_in)

optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)

iteration = 0
epochs = int(opt.iterations / len(trainset))

# keep track of training progress
train_log = open('log_repro_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)

# generate grid of target reprojection pixel positions
prediction_grid = torch.zeros((2, 
	math.ceil(5000 / network.OUTPUT_SUBSAMPLE),  # 5000px is max limit of image size, increase if needed
	math.ceil(5000 / network.OUTPUT_SUBSAMPLE)))

for x in range(0, prediction_grid.size(2)):
	for y in range(0, prediction_grid.size(1)):
		prediction_grid[0, y, x] = x * network.OUTPUT_SUBSAMPLE + network.OUTPUT_SUBSAMPLE / 2
		prediction_grid[1, y, x] = y * network.OUTPUT_SUBSAMPLE + network.OUTPUT_SUBSAMPLE / 2

prediction_grid = prediction_grid.cuda()

for epoch in range(epochs):	

	print("=== Epoch: %d ======================================" % epoch)

	for image, gt_pose, gt_coords, focal_length, file in trainset_loader:

		start_time = time.time()

		image = image.cuda()
		padX, padY, image = util.random_shift(image, network.OUTPUT_SUBSAMPLE / 2)
	
		prediction, neural_guidance = network(image) 
		# neural guidance is ignored / not trained during initlization

		# apply random shift to the ground truth reprojection positions as well
		prediction_grid_pad = prediction_grid[:,0:prediction.size(2),0:prediction.size(3)].clone()
		prediction_grid_pad = prediction_grid_pad.view(2, -1)

		prediction_grid_pad[0] -= padX
		prediction_grid_pad[1] -= padY

		# create camera calibartion matrix
		focal_length = float(focal_length[0])
		cam_mat = torch.eye(3)
		cam_mat[0, 0] = focal_length
		cam_mat[1, 1] = focal_length
		cam_mat[0, 2] = image.size(3) / 2
		cam_mat[1, 2] = image.size(2) / 2
		cam_mat = cam_mat.cuda()

		# predicted scene coordinates to homogeneous coordinates
		ones = torch.ones((prediction.size(0), 1, prediction.size(2), prediction.size(3)))
		ones = ones.cuda()
		prediction = torch.cat((prediction, ones), 1)

		gt_pose = gt_pose[0].inverse()[0:3,:]
		gt_pose = gt_pose.cuda()

		# scene coordinate to camera coordinate 
		prediction = prediction[0].view(4, -1)
		eye = torch.mm(gt_pose, prediction)

		# image reprojection
		px = torch.mm(cam_mat, eye)
		px[2].clamp_(min=0.1) #avoid division by zero
		px = px[0:2] / px[2]

		# reprojection error
		px = px - prediction_grid_pad
		px = px.norm(2, 0)
		px = px.clamp(0, opt.hardclamp) # reprojection error beyond 100px is not useful

		loss_l1 = px[px <= opt.softclamp]
		loss_sqrt = px[px > opt.softclamp]
		loss_sqrt = torch.sqrt(opt.softclamp * loss_sqrt)

		robust_loss = (loss_l1.sum() + loss_sqrt.sum()) / float(px.size(0))

		robust_loss.backward()	# calculate gradients (pytorch autograd)
		optimizer.step()		# update all network parameters
		optimizer.zero_grad()
		
		print('Iteration: %6d, Loss: %.1f, Time: %.2fs' % (iteration, robust_loss, time.time()-start_time), flush=True)
		train_log.write('%d %f\n' % (iteration, robust_loss))

		iteration = iteration + 1

	print('Saving snapshot of the network to %s.' % opt.network_out)
	torch.save(network.state_dict(), opt.network_out)
	

print('Done without errors.')
train_log.close()