from trainer.gan_trainer import to_variable, to_numpy
from data_loader import data_loaders
import model.cycle_gan_complex as gan_complex

import numpy as np
import cv2 as cv
import torch

def convert_svhn(old_svhn):
    new_svhn = np.zeros((32, 32, 3))
    for dim in range(0, 3):
        new_svhn[:,:,dim] = old_svhn[dim,:,:]
    return new_svhn

checkpoint_number = 15

generator_ab = gan_complex.ResnetGenerator(n_input_channels=1, n_output_channels=3)
generator_ba = gan_complex.ResnetGenerator(n_input_channels=3, n_output_channels=1)

generator_ab.load_state_dict(torch.load('./data/models/{}__genAB.pkl'.format(checkpoint_number)))
generator_ba.load_state_dict(torch.load('./data/models/{}__genBA.pkl'.format(checkpoint_number)))

generator_ab.cuda()
generator_ba.cuda()

svhn_loader = data_loaders.SVHNDataLoaderValidation(data_dir='./data/svhn/', batch_size=30, shuffle=True,
                                                    validation_split=0.0, num_workers=1, img_size=32, training=True)
mnist_loader = data_loaders.MnistDataLoader(data_dir='./data/mnist/', batch_size=30, shuffle=True, validation_split=0.0,
                                           num_workers=1, img_size=32, training=True)

svhn_iterator, mnist_iterator = iter(svhn_loader), iter(mnist_loader)
svhn_data, _ = next(svhn_iterator)
mnist_data, _ = next(mnist_iterator)

svhn_data, mnist_data = to_variable(svhn_data), to_variable(mnist_data)
generator_ab.eval()
generator_ba.eval()
fake_svhn, fake_mnist = generator_ab(mnist_data), generator_ba(svhn_data)

ii = 1
for real_svhn, real_mnist, gen_svhn, gen_mnist in zip(svhn_data, mnist_data, fake_svhn, fake_mnist):
    # Convert each sample into a numpy array in order to display with opencv
    real_svhn, real_mnist = to_numpy(real_svhn), to_numpy(real_mnist)
    gen_svhn, gen_mnist = to_numpy(gen_svhn), to_numpy(gen_mnist)

    # Rearrange indices in order to put the channel index at the end
    real_svhn, gen_svhn = convert_svhn(real_svhn), convert_svhn(gen_svhn)

    # Convert SVHN samples into BGR from RGB
    # real_svhn, gen_svhn = cv.cvtColor(real_svhn, cv.COLOR_BGR2RGB), cv.cvtColor(gen_svhn, cv.COLOR_BGR2RGB)

    cv.imshow('FAKE MNIST', cv.bitwise_not(gen_mnist.squeeze(0)))
    cv.imshow('REAL MNIST', real_mnist.squeeze(0))
    cv.imshow('FAKE SVHN', (gen_svhn * 0.50) + 0.50)
    cv.imshow('REAL SVHN', real_svhn)

    cv.waitKey(0)