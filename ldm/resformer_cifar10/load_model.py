import torch, torchvision
import numpy as np
from matplotlib import pyplot as plt

from resformer import *
from utils import train, validate

print(torch.load("resformer_cifar10.ckpt"))