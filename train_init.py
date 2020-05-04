import torch
import torch.optim as optim

import time
import argparse

from dataset import CamLocDataset
from network import Network
import util

parser = argparse.ArgumentParser(
	description='Train scene coordinate regression using target scene coordinates.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder')

parser.add_argument('network', help='output file name for the network')

parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, 
	help='learning rate')

parser.add_argument('--iterations', '-iter', type=int, default=500000,
	help='number of training iterations, i.e. numer of model updates')

parser.add_argument('--softclamp', '-sc', type=float, default=10, 
	help='robust square root loss after this threshold, in meters')

parser.add_argument('--hardclamp', '-hc', type=float, default=1000, 
	help='clamp loss with this threshold, in meters')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files, useful to separate different runs of a script')

opt = parser.parse_args()

trainset = CamLocDataset("./dataset/" + opt.scene + "/train")
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6)

print("Found %d training images for %s." % (len(trainset), opt.scene))

print("Calculating mean scene coordinate for the scene...")

mean = torch.zeros((3))
count = 0

for image, gt_pose, gt_coords, focal_length, file in trainset_loader:

	gt_coords = gt_coords[0]
	gt_coords = gt_coords.view(3, -1)

	coord_mask = gt_coords.abs().sum(0) > 0
	if coord_mask.sum() > 0:
		gt_coords = gt_coords[:, coord_mask]

	mean += gt_coords.median(1)[0]
	
mean /= len(trainset)

print("Done. Mean: %.2f, %.2f, %.2f\n" % (mean[0], mean[1], mean[2]))

# create network
network = Network(mean)
network = network.cuda()
network.train()

optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)

iteration = 0
epochs = int(opt.iterations / len(trainset))

# keep track of training progress
train_log = open('log_init_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)

for epoch in range(epochs):	

	print("=== Epoch: %d ======================================" % epoch)

	for image, gt_pose, gt_coords, focal_length, file in trainset_loader:

		start_time = time.time()

		gt_coords = gt_coords.cuda()
		image = image.cuda()

		#random shift as data augmentation
		padX, padY, image = util.random_shift(image, network.OUTPUT_SUBSAMPLE / 2)
	
		prediction, neural_guidance = network(image) 
		# neural guidance is ignored / not trained during initlization

		prediction = prediction.squeeze().view(3, -1)
		gt_coords = gt_coords.squeeze().view(3, -1)

		# mask out invalid coordinates (all zeros)
		coords_mask = gt_coords.abs().sum(0) != 0 

		if coords_mask.sum() == 0:
			print("Empty ground truth scene coordinates! Skip.")
			continue

		prediction = prediction[:,coords_mask]
		gt_coords = gt_coords[:,coords_mask]

		loss = torch.norm(prediction - gt_coords, dim=0)

		loss_mask = loss < opt.hardclamp
		loss = loss[loss_mask]

		# soft clamping of loss for stability
		loss_l1 = loss[loss <= opt.softclamp]
		loss_sqrt = loss[loss > opt.softclamp]
		loss_sqrt = torch.sqrt(opt.softclamp * loss_sqrt)

		robust_loss = (loss_l1.sum() + loss_sqrt.sum()) / float(loss.size(0))

		robust_loss.backward()	# calculate gradients (pytorch autograd)
		optimizer.step()		# update all network parameters
		optimizer.zero_grad()
		
		print('Iteration: %6d, Loss: %.1f, Time: %.2fs' % (iteration, robust_loss, time.time()-start_time), flush=True)
		train_log.write('%d %f\n' % (iteration, robust_loss))

		iteration = iteration + 1

	print('Saving snapshot of the network to %s.' % opt.network)
	torch.save(network.state_dict(), opt.network)
	

print('Done without errors.')
train_log.close()
