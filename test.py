import torch
import torch.nn.functional as F
import numpy as np
import cv2

import ngdsac

import time
import argparse
import math

from dataset import CamLocDataset
from network import Network

parser = argparse.ArgumentParser(
	description='Test a trained network on a specific scene.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder')

parser.add_argument('network', help='file name of a network trained for the scene')

parser.add_argument('--hypotheses', '-hyps', type=int, default=256, 
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10, 
	help='inlier threshold in pixels')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
	help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--inlierbeta', '-ib', type=float, default=0.5, 
	help='beta parameter of the soft inlier count; controls the softness of the sigmoid; lower means softer')

parser.add_argument('--maxreprojection', '-maxr', type=float, default=100, 
	help='maximum reprojection error; reprojection error is clamped to this value for stability')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files, useful to separate different runs of a script')

parser.add_argument('--uniform', '-u', action='store_true', 
	help='disable neural-guidance and sample data points uniformely; corresponds to a DSAC model')

opt = parser.parse_args()

# setup dataset
testset = CamLocDataset("./dataset/" + opt.scene + "/test", training=False)
testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=6)

# load network
network = Network(torch.zeros((3)))
network.load_state_dict(torch.load(opt.network))
network = network.cuda()
network.eval()

test_log = open('test_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)
pose_log = open('poses_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)

print('Test images found: ', len(testset))

# keep track of rotation and translation errors for calculation of the mean error
rErrs = []
tErrs = []
avg_time = 0

with torch.no_grad():	

	for image, gt_pose, init, focal_length, file in testset_loader:

		focal_length = float(focal_length[0])
		file = file[0].split('/')[-1] # remove path from file name
		gt_pose = gt_pose[0]
		image = image.cuda()

		start_time = time.time()

		# predict scene coordinates and neural guidance
		scene_coordinates, log_ng = network(image)
		
		scene_coordinates = scene_coordinates.cpu()
		neural_guidance = torch.exp(log_ng).cpu()

		if opt.uniform: 
			# overwrite neural guidance with uniform probabilities
			neural_guidance.fill_(1 / (log_ng.shape[2]*log_ng.shape[3]))

		out_pose = torch.zeros((4, 4))

		ngdsac.forward(
			scene_coordinates, 
			neural_guidance,
			out_pose, 
			opt.hypotheses, 
			opt.threshold,
			focal_length, 
			float(image.size(3) / 2), #principal point assumed in image center
			float(image.size(2) / 2), 
			opt.inlieralpha,
			opt.inlierbeta,
			opt.maxreprojection,
			network.OUTPUT_SUBSAMPLE)

		avg_time += time.time()-start_time

		# calculate pose errors
		t_err = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))

		gt_R = gt_pose[0:3,0:3].numpy()
		out_R = out_pose[0:3,0:3].numpy()

		r_err = np.matmul(out_R, np.transpose(gt_R))
		r_err = cv2.Rodrigues(r_err)[0]
		r_err = np.linalg.norm(r_err) * 180 / math.pi

		print("\nRotation Error: %.2fdeg, Translation Error: %.1fcm" % (r_err, t_err*100))

		rErrs.append(r_err)
		tErrs.append(t_err * 100)

		# write estimated pose to pose file
		out_pose = out_pose.inverse()

		t = out_pose[0:3, 3]

		# rotation to axis angle
		rot, _ = cv2.Rodrigues(out_pose[0:3,0:3].numpy())
		angle = np.linalg.norm(rot)
		axis = rot / angle

		# axis angle to quaternion
		q_w = math.cos(angle * 0.5)
		q_xyz = math.sin(angle * 0.5) * axis

		pose_log.write("%s %f %f %f %f %f %f %f %f %f\n" % (
			file,
			q_w, q_xyz[0], q_xyz[1], q_xyz[2],
			t[0], t[1], t[2],
			r_err, t_err))	

median_idx = int(len(rErrs)/2)
tErrs.sort()
rErrs.sort()
avg_time /= len(rErrs)

print("\n===================================================")
print("\nTest complete.")

print("\nMedian Error: %.1fdeg, %.1fcm" % (rErrs[median_idx], tErrs[median_idx]))
print("Avg. processing time: %4.1fms" % (avg_time * 1000))
test_log.write('%f %f %f\n' % (rErrs[median_idx], tErrs[median_idx], avg_time))

test_log.close()
pose_log.close()