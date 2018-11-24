import torch.nn.functional as func
import torch

def nll_loss(output, target):
    return func.nll_loss(output, target)

def discriminator_loss(output, true_label):
    """
    :param output:
    :param true_label:
    :return:
    """
    return torch.Tensor.mean((output - true_label)**2)
