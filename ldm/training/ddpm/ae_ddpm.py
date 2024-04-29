import torch
from typing import Any
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from .ddpm import DDPM

class AE_DDPM(DDPM):
    def __init__(self, ae_model, model, trainloader, testloader, config_dict, device='cuda:0'):
        self.ae_model = ae_model
        input_dim = self.ae_model.in_dim
        input_noise = torch.randn((1, input_dim))
        latent_dim = ae_model.encode(input_noise).shape
        # config.system.model.arch.model.in_dim = latent_dim[-1] * latent_dim[-2]
        self.model = model
        # model.in_dim = latent_dim[-1] * latent_dim[-2]
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        super(AE_DDPM, self).__init__(config_dict)
        self.save_hyperparameters()
        self.current_epoch = 0
        self.split_epoch = 30000
        self.loss_func = nn.MSELoss()
        self.optimizers = self.configure_optimizers()

    def ae_forward(self, batch, **kwargs):
        self.ae_model.to(batch.device)
        output = self.ae_model(batch)
        #debug
        # print("AEDDPM output.shape:", output.shape)
        #debug
        loss = self.loss_func(batch, output, **kwargs)
        # self.log('epoch', self.current_epoch)
        # self.log('ae_loss', loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def train(self, num_epochs, print_info=True):
        for _ in range(num_epochs):
            self.current_epoch += 1
            epoch_train_losses, epoch_valid_acc = [], []
            for i, data in tqdm(enumerate(self.trainloader), total=len(self.trainloader)):
                data = data.to(self.device)
                batch_train_loss = self.training_step(data)['loss'].detach().cpu().numpy()
                batch_valid_acc = self.validation_step(data)
                epoch_train_losses.append(batch_train_loss)
                epoch_valid_acc.append(batch_valid_acc)
            if print_info:
                print("epoch: ", self.current_epoch, 
                      ", train loss:", np.mean(epoch_train_losses),
                      ", validation acc (best):", np.mean([acc['best_g_acc'] for acc in epoch_valid_acc]),
                      ", validation acc (mean):", np.mean([acc['mean_g_acc'] for acc in epoch_valid_acc])
                    )

            

    def training_step(self, batch, **kwargs):
        ddpm_optimizer, ae_optimizer = self.optimizers
        if  self.current_epoch < self.split_epoch:
            loss = self.ae_forward(batch, **kwargs)
            ae_optimizer.zero_grad()
            # self.manual_backward(loss)
            loss.backward()
            ae_optimizer.step()
        else:
            loss = self.forward(batch, **kwargs)
            ddpm_optimizer.zero_grad()
            # self.manual_backward(loss)
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
            latent = self.ae_model.encode(good_param)
            # print("latent shape:{}".format(latent.shape))
            ae_params = self.ae_model.decode(latent)
            # print("ae params shape:{}".format(ae_params.shape))
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
        ddpm_params = self.model.parameters()
        self.ddpm_optimizer = torch.optim.AdamW(params=ddpm_params, lr=1e-3)
        self.ae_optimizer = torch.optim.AdamW(params=ae_params, lr=1e-3)
        # if 'lr_scheduler' in self.train_cfg and self.train_cfg.lr_scheduler is not None:
        #     self.lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler)
        return self.ddpm_optimizer, self.ae_optimizer