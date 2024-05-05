import torch
from typing import Any
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from .ddpm import DDPM

class AE_DDPM(DDPM):
    def __init__(self, ae_model, all_target_models, train_archs, test_archs, all_train_layers, all_trainloaders, testloader, config_dict, device='cuda:0'):
        self.ae_model = ae_model
        self.all_target_models = all_target_models
        self.train_archs = train_archs
        self.test_archs = test_archs
        self.all_train_layers = all_train_layers
        self.cur_model_arch = self.train_archs[0]
        self.all_trainloaders = all_trainloaders
        self.testloader = testloader
        self.device = device
        super(AE_DDPM, self).__init__(config_dict)
        self.current_epoch = 0
        self.split_epoch = 30000
        self.loss_func = nn.MSELoss()
        self.optimizers = self.configure_optimizers()
        self.save_hyperparameters()

    def save_hyperparameters(self):
        return torch.save(self.ae_model.state_dict, "ae_model_"+str(self.current_epoch)+".pth")

    def ae_forward(self, batch, **kwargs):
        self.ae_model.to(batch.device)
        output = self.ae_model(batch)
        loss = self.loss_func(batch, output, **kwargs)
        return loss

    def train(self, num_epochs, print_info=True):
        for _ in range(num_epochs):
            self.current_epoch += 1
            for arch_name in self.train_archs:
                self.cur_model_arch = arch_name
                #debug
                self.ddpm_optimizer = torch.optim.AdamW(params=self.get_cur_model().parameters(), lr=1e-3)
                print('arch_name:', arch_name)
                # print(self.get_cur_model())
                #debug
                epoch_arch_train_losses, epoch_arch_valid_acc = [], []
                trainloader = self.all_trainloaders[arch_name]
                for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
                    data = data.to(self.device)
                    batch_train_loss = self.training_step(data)['loss'].detach().cpu().numpy()
                    batch_valid_acc = self.validation_step(data)
                    epoch_arch_train_losses.append(batch_train_loss)
                    epoch_arch_valid_acc.append(batch_valid_acc)
                if print_info:
                    print("epoch: ", self.current_epoch, 
                        ", NN arch:", arch_name,
                        ", train loss:", np.mean(epoch_arch_train_losses),
                        ", validation acc (best):", np.mean([acc['best_g_acc'] for acc in epoch_arch_valid_acc]),
                        ", validation acc (mean):", np.mean([acc['mean_g_acc'] for acc in epoch_arch_valid_acc])
                        )


    def training_step(self, batch, **kwargs):
        ddpm_optimizers, ae_optimizer = self.optimizers
        ddpm_optimizer = ddpm_optimizers[self.cur_model_arch]
        if  self.current_epoch < self.split_epoch:
            loss = self.ae_forward(batch, **kwargs)
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
        else:
            loss = self.forward(batch, **kwargs)
            ddpm_optimizer.zero_grad()
            loss.backward()
            ddpm_optimizer.step()

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
        return {'loss': loss}

    def pre_process(self, batch):
        latent =  self.ae_model.encode(batch)
        self.latent_shape = latent.shape[-2:]
        return latent

    def post_process(self, outputs):
        outputs = outputs.reshape(-1, *self.latent_shape)
        return self.ae_model.decode(outputs)

    def validation_step(self, batch, **kwargs: Any):
        if self.current_epoch < self.split_epoch:
            # todo
            # good_param = batch[:10]
            good_param = batch
            input_accs = []
            for i, param in enumerate(good_param):
                acc, test_loss, output_list = self.test_g_model(param)
                input_accs.append(acc)
            # print("input model accuracy:{}".format(input_accs))
            # """
            # AE reconstruction parameters
            # """
            # print('---------------------------------')
            # print('Test the AE model')
            ae_rec_accs = []
            # latent = self.ae_model.encode(good_param)
            # # print("latent shape:{}".format(latent.shape))
            # ae_params = self.ae_model.decode(latent)
            ae_params = self.ae_model(good_param)
            ae_params = self.ae_model(torch.rand(good_param.shape, dtype=good_param.dtype, device=good_param.device))
            # ae_params = -10*torch.rand(good_param.shape, dtype=good_param.dtype, device=good_param.device)
            ae_params = ae_params.cpu()
            for i, param in enumerate(ae_params):
                param = param.to(batch.device)
                acc, test_loss, output_list = self.test_g_model(param)
                ae_rec_accs.append(acc)

            best_ae = max(ae_rec_accs)
            # print(f'AE reconstruction models accuracy:{ae_rec_accs}')
            # print(f'AE reconstruction models best accuracy:{best_ae}')
            # print('---------------------------------')
            return {"best_g_acc": best_ae, "mean_g_acc": np.mean(ae_rec_accs)}
        else:
            dict = super(AE_DDPM, self).validation_step(batch, **kwargs)
            return dict

    def configure_optimizers(self, **kwargs):
        ae_params = self.ae_model.parameters()
        self.ddpm_optimizers = {
            arch_name: torch.optim.AdamW(
                params=self.all_target_models[arch_name].parameters(), lr=1e-3)
            for arch_name in self.all_target_models.keys()
        }
        self.ae_optimizer = torch.optim.AdamW(params=ae_params, lr=1e-3)
        return self.ddpm_optimizers, self.ae_optimizer