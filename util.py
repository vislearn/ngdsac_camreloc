import torch.nn as nn
import random

def random_shift(image, max_shift):
	''' Randomly shift the input image via zero padding in x and y. '''

	padX = random.randint(-max_shift, max_shift)
	padY = random.randint(-max_shift, max_shift)
	pad = nn.ZeroPad2d((padX, -padX, padY, -padY))

	return padX, padY, pad(image)