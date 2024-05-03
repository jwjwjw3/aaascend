import sys
import torch, torchvision
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append("../..")
from training.ddpm.ae_ddpm import AE_DDPM
from training.dataset.parametersDataset import ResNetParamsDataset
from diffResFormer.BNResFormer import BNResFormer

sys.path.append("../../../resnet_params_data/utils")
from ResNet import *


def train():
    config_dict = {
        "beta_schedule": {
            "start": 1e-4,
            "end": 2e-2,
            "schedule": "linear",
            "n_timestep": 50,
        },
        "model_mean_type": "eps",
        "model_var_type": "fixedlarge",
        "loss_type": "mse",
    }

    # raw_data = torch.load("./_scratch_folder/ResNet18_2222_4-200-data.pt")
    # params_dataset = raw_data['pdata']
    # dataset_train = ParametersDatset(params_dataset, 180, split='train')
    dataset_train = ResNetParamsDataset(model_arch_name="ResNet18_2222_4", 
        train_layer=['blks.7.bn1.weight', 'blks.7.bn1.bias', 'blks.7.bn2.weight', 'blks.7.bn2.bias'], 
        k=180, split='train')
    train_batch_size = 60
    test_batch_size = 1000
    params_train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=train_batch_size,
        shuffle=True
    )

    cifar10_valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(28, 28), antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cifar10_test_dataset = torchvision.datasets.CIFAR10(
        root = "../../../resnet_params_data/cifar10_dataset", 
        train = False,
        transform = cifar10_valid_transforms,
        download=False,
    )
    cifar10_test_loader = torch.utils.data.DataLoader(
        cifar10_test_dataset, 
        batch_size=test_batch_size,
        shuffle=False,
        sampler=SubsetRandomSampler([i for i in range(100)]),
    )

    target_model = ResNet18_2222_4() 
    target_model.load_state_dict(torch.load("../../../resnet_params_data/models/ResNet18_2222_4/ResNet18_2222_4_seed_190.ckpt"))
    trainer = AE_DDPM(
                ae_model=BNResFormer(input_dim=1, output_dim=1, depth=5),
                model=target_model,
                trainloader = params_train_loader,
                testloader = cifar10_test_loader,
                config_dict=config_dict)
    
    # trainer.train_layer = raw_data['train_layer']
    trainer.train_layer = dataset_train.train_layer

    trainer.train(100)


if __name__ == "__main__":
    train()