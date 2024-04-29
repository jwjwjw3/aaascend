from torchvision.datasets.vision import VisionDataset

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