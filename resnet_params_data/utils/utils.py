import os
import torch, torchvision
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from ResNet import *

def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc



def resnet_train(model_name, random_seed, dataset_folder, model_folder, train_batch_size=128, test_batch_size=100, 
                epochs=50, device="cuda:0"):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.makedirs(model_folder, exist_ok=True) 
    model = eval(model_name+"()")
    model.train()
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # load dataset
    train_trfs = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(32, 32), antialias=True),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    valid_trfs = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(32, 32), antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_train = CIFAR10(root=dataset_folder, train=True, transform=train_trfs, download=True)
    dataset_valid = CIFAR10(root=dataset_folder, train=False, transform=valid_trfs, download=True)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=test_batch_size, shuffle=False)
    # perform training
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
            torch.save(model.state_dict(), 
                os.path.join(model_folder, model_name+"_seed_"+str(random_seed)+".ckpt"))
    # save train logs 
    single_gpu_resnet_cifar10_results = {
        "train_loss": train_loss, "valid_loss": valid_loss,
        "train_acc": train_acc, "valid_acc": valid_acc}
    torch.save(single_gpu_resnet_cifar10_results, 
        os.path.join(model_folder, "trainlog_"+model_name+"_seed_"+str(random_seed)+".torch"))