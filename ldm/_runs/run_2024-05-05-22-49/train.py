import sys
import torch, torchvision
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append("../..")
from training.ddpm.ae_ddpm import AE_DDPM
from training.dataset.parametersDataset import ResNetParamsDataset
from training.dataset.configs import ALL_ARCH_TRAINABLE_BN_LAYERS
from diffResFormer.BNResFormer import BNResFormer

sys.path.append("../../../resnet_params_data/utils")
from ResNet import *


DDPM_CONFIG_DICT = {
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

train_batch_size = 180
test_batch_size = 1000


def train():
    train_archs = [
        "ResNet12_2111_8", "ResNet14_222_8", 
        "ResNet14_2211_8", "ResNet16_2221_8", 
        "ResNet18_2222_8", "ResNet34_3463_8",
    ]
    test_archs = [
        "ResNet10_22_8", "ResNet10_1111_8"
    ]
    train_archs.reverse()
    test_archs.reverse()
    all_params_datasets = {
        arch_name: ResNetParamsDataset(model_arch_name=arch_name, 
                    train_layer=ALL_ARCH_TRAINABLE_BN_LAYERS[arch_name][1], k=180, split='train')
        for arch_name in list(ALL_ARCH_TRAINABLE_BN_LAYERS.keys())
    }
    all_params_loaders = {
        arch_name: torch.utils.data.DataLoader(all_params_datasets[arch_name], 
                        batch_size=train_batch_size, shuffle=True)
        for arch_name in list(ALL_ARCH_TRAINABLE_BN_LAYERS.keys())
    }
    all_target_models = {arch_name: all_params_datasets[arch_name].model for arch_name in all_params_datasets.keys()}
    all_train_layers = {arch_name: all_params_datasets[arch_name].train_layer for arch_name in all_params_datasets.keys()}

    cifar10_valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(32, 32), antialias=True),
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

    trainer = AE_DDPM(
                ae_model=BNResFormer(input_dim=1, output_dim=1, input_mlp_dims=[4, 8], embed_dim=8,
                    depth=4, num_heads=4, mlp_ratio=4., qkv_bias=True, in_out_bias=True),
                all_target_models=all_target_models,
                train_archs=train_archs,
                test_archs=test_archs,
                train_epoch_foreach_arch_bool=True,
                all_train_layers=all_train_layers,
                all_trainloaders=all_params_loaders,
                testloader=cifar10_test_loader,
                config_dict=DDPM_CONFIG_DICT)

    trainer.train(50)
    trainer.save_hyperparameters()


if __name__ == "__main__":
    train()