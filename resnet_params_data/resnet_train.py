import sys
import torch, torchvision

sys.path.append("./resnet")

from resnet import ResNet18
from utils import train, validate


random_seed = 100
model_name = "ResNet18"
train_batch_size = 128
test_batch_size = 100
epochs = 50
device = "cuda:0"


torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

model = ResNet18()

model.train()
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

cifar10_train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32), antialias=True),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
cifar10_valid_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32), antialias=True),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset_train = torchvision.datasets.CIFAR10(
    root = "./cifar10_dataset", 
    train = True,
    transform = cifar10_train_transforms,
    download=True,
)
dataset_test = torchvision.datasets.CIFAR10(
    root = "./cifar10_dataset", 
    train = False,
    transform = cifar10_valid_transforms,
    download=True,
)

train_loader = torch.utils.data.DataLoader(
    dataset_train, 
    batch_size=train_batch_size,
    shuffle=True
)
valid_loader = torch.utils.data.DataLoader(
    dataset_test, 
    batch_size=test_batch_size,
    shuffle=False
)


train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, device)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    if len(valid_acc) == 0 or max(valid_acc) == valid_epoch_acc:
        print("Saving current best model...")
        torch.save(model.state_dict(), model_name+"_seed_"+str(random_seed)+"_cp.ckpt")

single_gpu_resnet_cifar10_results = {
    "train_acc": train_acc,
    "valid_acc": valid_acc,
    "train_loss": train_loss,
    "valid_loss": valid_loss,
}

torch.save(single_gpu_resnet_cifar10_results, "trainlog_"+model_name+"_seed_"+str(random_seed)+"_cp.torch")