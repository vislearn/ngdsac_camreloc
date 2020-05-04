import torch
import torch.optim as optim

import argparse
import time
import random

import ngdsac
import util

from dataset import CamLocDataset
from network import Network

parser = argparse.ArgumentParser(
	description='Train scene coordinate regression and neural guidance in an end-to-end fashion.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder')

parser.add_argument('network_in', help='file name of a network initialized for the scene')

parser.add_argument('network_out', help='output file name for the new network')

parser.add_argument('--hypotheses', '-hyps', type=int, default=16, 
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10, 
	help='inlier threshold in pixels')

parser.add_argument('--inlieralpha', '-ia', type=float, default=10, 
	help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--inlierbeta', '-ib', type=float, default=0.5, 
	help='beta parameter of the soft inlier count; controls the softness of the sigmoid; lower means softer')

parser.add_argument('--learningrate', '-lr', type=float, default=0.000001, 
	help='learning rate')

parser.add_argument('--iterations', '-it', type=int, default=200000, 
	help='number of training iterations, i.e. network parameter updates')

parser.add_argument('--weightrot', '-wr', type=float, default=1.0, 
	help='weight of rotation part of pose loss')

parser.add_argument('--weighttrans', '-wt', type=float, default=1.0, 
	help='weight of translation part of pose loss')

parser.add_argument('--maxreprojection', '-maxr', type=float, default=100, 
	help='maximum reprojection error; reprojection error is clamped to this value for stability')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files. Useful to separate different runs of the program')

parser.add_argument('--samples', '-s', type=int, default=2, 
	help='number of samples per training image to approximate the loss expectation')

parser.add_argument('--uniform', '-u', action='store_true', 
	help='disable neural-guidance and sample data points uniformely; corresponds to a DSAC model')

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
train_log = open('log_e2e_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)

for epoch in range(epochs):	

	print("=== Epoch: %7d ======================================" % epoch)

	for image, pose, init, focal_length, file in trainset_loader:

		start_time = time.time()

		focal_length = float(focal_length[0])
		pose = pose[0]
		image = image.cuda()

		#random shift as data augmentation
		padX, padY, image = util.random_shift(image, network.OUTPUT_SUBSAMPLE / 2)

		# predict scene coordinates and neural guidance
		scene_coordinates, log_ng = network(image)
		neural_guidance = torch.exp(log_ng).cpu()

		if opt.uniform: 
			# overwrite neural guidance with uniform probabilities
			neural_guidance.fill_(1 / (log_ng.shape[2]*log_ng.shape[3]))

		scene_coordinate_gradients = torch.zeros(scene_coordinates.size())
		log_ng_gradients = torch.zeros(log_ng.size())

		# we run the pipeline multiple times per image to approximate the expectation objective
		ng_gradient_samples = []
		loss_samples = []

		print("--- Start Sampling --------------------------------------")

		for sample in range(opt.samples):

			local_scene_coordinate_gradients = torch.zeros(scene_coordinate_gradients.size())
			local_log_ng_gradients = torch.zeros(log_ng_gradients.size())

			local_loss = ngdsac.backward(
				scene_coordinates.cpu(), 
				local_scene_coordinate_gradients,
				neural_guidance,
				local_log_ng_gradients,
				pose, 
				padX, 
				padY, 
				opt.hypotheses, 
				opt.threshold,
				focal_length, 
				float(image.size(3) / 2), #principal point assumed in image center
				float(image.size(2) / 2),
				opt.weightrot,
				opt.weighttrans,
				opt.inlieralpha,
				opt.inlierbeta,
				opt.maxreprojection,
				network.OUTPUT_SUBSAMPLE,
				sample) #used for randomizing the seed

			scene_coordinate_gradients += local_scene_coordinate_gradients

			loss_samples.append(local_loss)
			ng_gradient_samples.append(local_log_ng_gradients)

			print('')

		print("---------------------------------------------------------")

		# baseline is mean loss over samples
		baseline = sum(loss_samples) / opt.samples
		
		# substract baseline and calculte gradients
		for i, l in enumerate(loss_samples):
			log_ng_gradients += ng_gradient_samples[i] * (l - baseline)
		
		log_ng_gradients /= opt.samples
		scene_coordinate_gradients /= opt.samples
	
		if opt.uniform:
			#if neural guidance is disabled only propagate scene coordinate gradients
			torch.autograd.backward((scene_coordinates), (scene_coordinate_gradients.cuda()))
		else:
			#default case: propage scene coordinate and neural guidance gradients
			torch.autograd.backward((scene_coordinates, log_ng), (scene_coordinate_gradients.cuda(), log_ng_gradients.cuda()))
	
		# update network parameters
		optimizer.step()
		optimizer.zero_grad()
		
		end_time = time.time()-start_time
		print('Iteration: %6d, Loss: %.2f, Time: %.2fs \n' % (iteration, baseline, end_time), flush=True)

		train_log.write('%d %f\n' % (iteration, baseline))
		iteration = iteration + 1

	print('Saving snapshot of the network to %s.' % opt.network_out)
	torch.save(network.state_dict(), opt.network_out)

print('Done without errors.')
train_log.close()
