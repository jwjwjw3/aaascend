import torch, torchvision
from models import ResNet34
from resnet_utils import train, validate

train_batch_size = 64
test_batch_size = 100
epochs = 50
device = "cuda:0"


cifar10_train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(28, 28), antialias=True),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar10_valid_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(28, 28), antialias=True),
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

model = ResNet34()
# model.eval()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Loss function.
criterion = torch.nn.CrossEntropyLoss()

model = model.to(device)

train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# Start the training.
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(
        model, 
        train_loader, 
        optimizer, 
        criterion,
        device
    )
    valid_epoch_loss, valid_epoch_acc = validate(
        model, 
        valid_loader, 
        criterion,
        device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)

    single_gpu_resnet_cifar10_results = {
    "train_acc": train_acc,
    "valid_acc": valid_acc,
    "train_loss": train_loss,
    "valid_loss": valid_loss,
}

torch.save(model.state_dict(), "./_scratch_folder/resnet34_cifar10.ckpt")