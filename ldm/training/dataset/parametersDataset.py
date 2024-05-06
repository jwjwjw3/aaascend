import os, sys
import torch
from torchvision.datasets.vision import VisionDataset
RESNET_UTILS_PATH = os.path.join(os.path.dirname(__file__), '../../../resnet_params_data/utils')
sys.path.append(RESNET_UTILS_PATH)
from ResNet import *

class ParametersDatset(VisionDataset):
    def __init__(self, batch, k, split='train', transform=None, target_transform=None):
        super(ParametersDatset, self).__init__(root=None, transform=transform, target_transform=target_transform)
        if split  == 'train':
            self.data = batch[:k]
        else:
            self.data = batch[:k]
        # data is a tensor list which is the parameters of the model

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        return len(self.data)


class ResNetParamsDataset(VisionDataset):
    def __init__(self, model_arch_name, train_layer, k, 
            split='train', transform=None, target_transform=None, test_model_indices=[i for i in range(180, 200)]):
        # exmaple: model_arch_name="ResNet18_2222_4", 
        # train_layer=['blks.7.bn1.weight', 'blks.7.bn1.bias', 'blks.7.bn2.weight', 'blks.7.bn2.bias']
        super(ResNetParamsDataset, self).__init__(root=None, transform=transform, target_transform=target_transform)
        # load default neural network model
        self.load_models_by_indices(model_arch_name=model_arch_name, indices=test_model_indices)
        # configure trainable layer names list
        self.train_layer = train_layer
        # load all trainable layers' data
        self.data = self.load_params_data(model_arch_name=model_arch_name, train_layer=train_layer)
        # perform train-test split
        if split  == 'train':
            self.data = self.data[:k]
        else:
            self.data = self.data[:k]

    def load_params_data(self, model_arch_name, train_layer):
        resnet_models_folder = os.path.join(RESNET_UTILS_PATH, "../models/")
        ckpt_files = sorted(
            [f for f in os.listdir(os.path.join(resnet_models_folder, model_arch_name)) if f[-5:] == ".ckpt"],
            key=lambda x: (len(x), x)
        )
        all_train_params = []
        for i in range(len(ckpt_files)):
            cur_model_params = torch.load(os.path.join(resnet_models_folder, model_arch_name, ckpt_files[i]), map_location='cpu')
            cur_train_params = torch.cat([cur_model_params[layer_name] for layer_name in train_layer])
            all_train_params.append(cur_train_params)
        all_train_params = torch.stack(all_train_params, dim=0)
        return all_train_params

    def load_models_by_indices(self, model_arch_name, indices):
        self.models = []
        for i in indices:
            cur_model = eval(model_arch_name+"()")
            model_ckpt_path = os.path.join(
                RESNET_UTILS_PATH,  "../models/", model_arch_name, model_arch_name+"_seed_"+str(i)+".ckpt"
            )
            cur_model.load_state_dict(torch.load(model_ckpt_path))
            self.models.append(cur_model)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        return len(self.data)