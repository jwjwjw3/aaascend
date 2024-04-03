import torch, torchvision
import numpy as np
from matplotlib import pyplot as plt

from resnet import resnet18, resnet50
from utils import train, validate

train_batch_size = 64
test_batch_size = 100
epochs = 50
device = "cuda:0"


cifar10_valid_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224), antialias=True),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset_test = torchvision.datasets.CIFAR10(
    root = "./cifar10_dataset", 
    train = False,
    transform = cifar10_valid_transforms,
    download=True,
)

valid_loader = torch.utils.data.DataLoader(
    dataset_test, 
    batch_size=test_batch_size,
    shuffle=False
)

model = resnet18(num_classes=10)
model.load_state_dict(torch.load("resnet18_cifar10.ckpt"))
model.eval()

# Loss function.
criterion = torch.nn.CrossEntropyLoss()

model = model.to(device)

train_acc, valid_acc = [], []
# Start the training.
valid_epoch_loss, valid_epoch_acc = validate(
    model, 
    valid_loader, 
    criterion,
    device
)
print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")