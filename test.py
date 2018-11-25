from trainer.gan_trainer import to_variable, to_numpy
from data_loader import data_loaders
from model import cycle_gan

from torchvision import transforms
import numpy as np
import cv2 as cv
import torch

def convert_svhn(old_svhn):
    new_svhn = np.zeros((32, 32, 3))
    for dim in range(0, 3):
        new_svhn[:,:,dim] = old_svhn[dim,:,:]
    return new_svhn

checkpoint_number = 99

generator_ab = cycle_gan.Generator(input_class_channels=1, output_class_channels=3, internal_channels=64)
generator_ba = cycle_gan.Generator(input_class_channels=3, output_class_channels=1, internal_channels=64)

generator_ab.load_state_dict(torch.load('./data/models/genAB-{}.pkl'.format(checkpoint_number)))
generator_ba.load_state_dict(torch.load('./data/models/genBA-{}.pkl'.format(checkpoint_number)))

generator_ab.cuda()
generator_ba.cuda()

svhn_loader = data_loaders.SVHNDataLoaderValidation(data_dir='./data/svhn/', batch_size=10, shuffle=True,
                                                    validation_split=0.0, num_workers=1, img_size=32, training=True)
mnist_loader = data_loaders.MnistDataLoader(data_dir='./data/mnist/', batch_size=10, shuffle=True, validation_split=0.0,
                                           num_workers=1, img_size=32, training=True)

svhn_iterator, mnist_iterator = iter(svhn_loader), iter(mnist_loader)
svhn_data, _ = next(svhn_iterator)
mnist_data, _ = next(mnist_iterator)

svhn_data, mnist_data = to_variable(svhn_data), to_variable(mnist_data)
fake_svhn, fake_mnist = generator_ab(mnist_data), generator_ba((svhn_data - 0.50)/0.50)

ii = 1
for real_svhn, real_mnist, gen_svhn, gen_mnist in zip(svhn_data, mnist_data, fake_svhn, fake_mnist):
    # Convert each sample into a numpy array in order to display with opencv
    real_svhn, real_mnist = to_numpy(real_svhn), to_numpy(real_mnist)
    gen_svhn, gen_mnist = to_numpy(gen_svhn), to_numpy(gen_mnist)

    # Rearrange indices in order to put the channel index at the end
    real_svhn, gen_svhn = convert_svhn(real_svhn), convert_svhn(gen_svhn)

    # Convert SVHN samples into BGR from RGB
    # real_svhn, gen_svhn = cv.cvtColor(real_svhn, cv.COLOR_BGR2RGB), cv.cvtColor(gen_svhn, cv.COLOR_BGR2RGB)

    cv.imshow('FAKE MNIST', gen_mnist.squeeze(0))
    cv.imshow('REAL MNIST', real_mnist.squeeze(0))
    cv.imshow('FAKE SVHN', (gen_svhn * 0.50) + 0.50)
    cv.imshow('REAL SVHN', real_svhn)

    print(real_svhn)

    cv.waitKey(0)

# svhn_data, mnist_data = to_variable(svhn_data[0]).unsqueeze(0), to_variable(mnist_data[0]).unsqueeze(0)
# real_svhn, real_mnist = to_numpy(svhn_data).squeeze(0), to_numpy(mnist_data).squeeze()
# real_svhn, real_mnist = np.swapaxes(real_svhn, 0, 2), np.swapaxes(real_mnist, 0, 1)
#
# fake_svhn, fake_mnist = to_numpy(generator_ab(mnist_data).squeeze(0)), to_numpy(generator_ba(svhn_data).squeeze())
# fake_svhn, fake_mnist = np.swapaxes(fake_svhn, 0, 2), np.swapaxes(fake_mnist, 0, 1)
#
# real_svhn, real_mnist = np.swapaxes(real_svhn, 0, 1), np.swapaxes(real_mnist, 0, 1)
# fake_svhn, fake_mnist = np.swapaxes(fake_svhn, 0, 1), np.swapaxes(fake_mnist, 0, 1)
#
# real_svhn, fake_svhn = cv.cvtColor(real_svhn, cv.COLOR_BGR2RGB), cv.cvtColor(fake_svhn, cv.COLOR_BGR2RGB)
#
# cv.imshow('FAKE SVHN', fake_svhn)
# cv.imshow('REAL SVHN', real_svhn)
# cv.imshow('FAKE MNIST', fake_mnist)
# cv.imshow('REAL MNIST', real_mnist)
#
# cv.waitKey(0)

