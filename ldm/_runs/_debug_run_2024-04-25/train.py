import sys
import torch, torchvision
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append("../..")

from training.ddpm.ae_ddpm import AE_DDPM
from training.dataset.parametersDataset import ParametersDatset
from diffResFormer.encoder import small
from diffResFormer.unet import AE_CNN_bottleneck

#debug
from models import *
#debug

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

    raw_data = torch.load("./_scratch_folder/data.pt")
    params_dataset = raw_data['pdata']
    dataset_train = ParametersDatset(params_dataset, 160, split='train')
    # dataset_test = ParametersDatset(params_dataset, 160, split='test')
    train_batch_size = 80
    test_batch_size = 1000
    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=train_batch_size,
        shuffle=True
    )
    # valid_loader = torch.utils.data.DataLoader(
    #     dataset_test, 
    #     batch_size=test_batch_size,
    #     shuffle=False
    # )

    cifar10_valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(28, 28), antialias=True),
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
        shuffle=False,
        sampler=SubsetRandomSampler([i for i in range(100)]),
    )

    target_model = ResNet34()
    target_model.load_state_dict(torch.load("./_scratch_folder/resnet34_cifar10.ckpt"))
    trainer = AE_DDPM(
                ae_model=small(in_dim=2048, input_noise_factor=0.1, latent_noise_factor=0.1), 
                # model=AE_CNN_bottleneck(in_dim=2048, in_channel=1, time_step=50, dec=None),
                model=target_model,
                trainloader =  train_loader,
                testloader = valid_loader,
                config_dict=config_dict)
    
    trainer.train_layer = raw_data['train_layer']

    # print(trainer.generate(batch=torch.rand(1, 1, 2048).cuda()).shape)

    # print(trainer.training_step(torch.rand(1, 2048, dtype=torch.float32).cuda(), batch_idx=0))
    trainer.train(100)


if __name__ == "__main__":
    train()