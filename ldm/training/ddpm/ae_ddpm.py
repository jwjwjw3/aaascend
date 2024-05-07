import os, time
import torch
from typing import Any
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from .ddpm import DDPM

class AE_DDPM(DDPM):
    def __init__(self, ae_model, all_target_models, train_archs, test_archs, train_epoch_foreach_arch_bool, all_train_layers, all_trainloaders, testloader, config_dict, device='cuda:0'):
        self.ae_model = ae_model
        self.all_target_models = all_target_models
        self.train_archs = train_archs
        self.test_archs = test_archs
        self.all_train_layers = all_train_layers
        self.cur_model_arch = self.train_archs[0]
        self.all_trainloaders = all_trainloaders
        self.testloader = testloader
        self.train_epoch_foreach_arch_bool = train_epoch_foreach_arch_bool  # True: each epoch has all archs; False: each arch completes all epochs one by one
        self.device = device
        self.ae_model.to(self.device)
        super(AE_DDPM, self).__init__(config_dict)
        self.current_epoch = 0
        self.split_epoch = 30000
        self.loss_func = nn.MSELoss()
        self.optimizers = self.configure_optimizers()
        self.save_hyperparameters()

    def save_hyperparameters(self, folder_path='./_analysis/ae_model_params'):
        os.makedirs(folder_path, exist_ok=True)
        return torch.save(self.ae_model.state_dict, os.path.join(folder_path, "ae_model_"+str(self.current_epoch)+".pth"))

    def ae_forward(self, batch, **kwargs):
        self.ae_model.to(batch.device)
        output = self.ae_model(batch)
        loss = self.loss_func(batch, output, **kwargs)
        return loss

    def train(self, num_epochs, print_info=True):
        if self.train_epoch_foreach_arch_bool:
            return self.train_epoch_foreach_arch(num_epochs=num_epochs, print_info=print_info)
        else:
            return self.train_arch_foreach_epoch(num_epochs=num_epochs, print_info=print_info)

    def train_epoch_foreach_arch(self, num_epochs, print_info):
        for _ in range(num_epochs):
            self.current_epoch += 1
            for arch_name in self.train_archs:
                self.cur_model_arch = arch_name
                # self.ddpm_optimizer = torch.optim.AdamW(params=self.get_cur_model().parameters(), lr=1e-3)
                epoch_arch_train_losses, epoch_arch_valid_acc = [], []
                trainloader = self.all_trainloaders[arch_name]
                for data in trainloader:
                    data = data.to(self.device)
                    batch_train_loss = self.training_step(data)['loss'].detach().cpu().numpy()
                    batch_valid_acc = self.validation_step(data)
                    epoch_arch_train_losses.append(batch_train_loss)
                    epoch_arch_valid_acc.append(batch_valid_acc)
                if print_info:
                    print("epoch:", self.current_epoch, 
                        ", NN arch:", arch_name,
                        ", train loss: {:.5}".format(np.mean(epoch_arch_train_losses)),
                        ", acc (best): {:.5}".format(np.mean([acc['best_g_acc'] for acc in epoch_arch_valid_acc])),
                        ", acc (mean): {:.5}".format(np.mean([acc['mean_g_acc'] for acc in epoch_arch_valid_acc]))
                        )
            self.test_acc_archs(print_info=print_info)
            self.save_hyperparameters()
            print("")

    def test_acc_archs(self, print_info):
        for arch_name in self.test_archs:
            self.cur_model_arch = arch_name
            epoch_test_arch_valid_acc = []
            trainloader = self.all_trainloaders[arch_name]
            for data in trainloader:
                data = data.to(self.device)
                batch_valid_acc = self.validation_step(data)
                epoch_test_arch_valid_acc.append(batch_valid_acc)
            if print_info:
                print("epoch:", self.current_epoch, 
                    ", test NN arch:", arch_name,
                    ", acc (best): {:.5}".format(np.mean([acc['best_g_acc'] for acc in epoch_test_arch_valid_acc])),
                    ", acc (mean): {:.5}".format(np.mean([acc['mean_g_acc'] for acc in epoch_test_arch_valid_acc]))
                    )


    def ae_g_model_speedtest(self, print_info):
        for arch_name in self.test_archs:
            self.cur_model_arch = arch_name
            epoch_test_arch_valid_acc = []
            trainloader = self.all_trainloaders[arch_name]
            i = 0
            start_time = time.time()
            for data in trainloader:
                data = data.to(self.device)
                _ = self.ae_forward(data)
                i += 1
            avg_forward_time = (time.time() - start_time) / i
            num_arch_total_params = sum([p.numel() for p in self.all_target_models[arch_name][0].parameters()])
            num_trainable_params = data.shape[1]
            # print(num_arch_total_params, num_trainable_params)
            avg_forward_time = avg_forward_time * num_arch_total_params / num_trainable_params
            if print_info:
                print("test NN arch:", arch_name,
                    ", mean forward time (s):", avg_forward_time
                    )

    def train_arch_foreach_epoch(self, num_epochs, print_info):
        for arch_name in self.train_archs:
            # self.ddpm_optimizer = torch.optim.AdamW(params=self.get_cur_model().parameters(), lr=1e-3)
            self.cur_model_arch = arch_name
            for _ in range(num_epochs):
                self.current_epoch += 1
                epoch_arch_train_losses, epoch_arch_valid_acc = [], []
                trainloader = self.all_trainloaders[arch_name]
                for data in trainloader:
                    data = data.to(self.device)
                    batch_train_loss = self.training_step(data)['loss'].detach().cpu().numpy()
                    batch_valid_acc = self.validation_step(data)
                    epoch_arch_train_losses.append(batch_train_loss)
                    epoch_arch_valid_acc.append(batch_valid_acc)
                if print_info:
                    print("epoch:", self.current_epoch, 
                        ", NN arch:", arch_name,
                        ", train loss: {:.5}".format(np.mean(epoch_arch_train_losses)),
                        ", validation acc (best): {:.5}".format(np.mean([acc['best_g_acc'] for acc in epoch_arch_valid_acc])),
                        ", validation acc (mean): {:.5}".format(np.mean([acc['mean_g_acc'] for acc in epoch_arch_valid_acc]))
                        )
            print("")


    def training_step(self, batch, **kwargs):
        ddpm_optimizers, ae_optimizer = self.optimizers
        # ddpm_optimizer = ddpm_optimizers[self.cur_model_arch]
        # if  self.current_epoch < self.split_epoch:
        loss = self.ae_forward(batch, **kwargs)
        ae_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()
        # else:
        #     loss = self.forward(batch, **kwargs)
        #     ddpm_optimizer.zero_grad()
        #     loss.backward()
        #     ddpm_optimizer.step()

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
            good_param = torch.unsqueeze(batch[0], 0)   # only one net's parameters
            # input_accs = []
            # for i, param in enumerate(good_param):
            #     acc, test_loss, output_list = self.test_g_model(param)
            #     input_accs.append(acc)
            # print("input model accuracy:{}".format(input_accs))
            # """
            # AE reconstruction parameters
            # """
            # print('---------------------------------')
            # print('Test the AE model')
            ae_rec_accs = []
            rand_param = torch.rand(good_param.shape, dtype=good_param.dtype, device=good_param.device)
            ae_params = self.ae_model(rand_param).cpu()
            for i, param in enumerate(ae_params):
                param = param.to(batch.device)
                all_accs, all_test_losses, all_output_lists = self.test_g_model(param)
                ae_rec_accs.extend(all_accs)

            return {"best_g_acc": max(ae_rec_accs), "mean_g_acc": np.mean(ae_rec_accs)}
        else:
            return super(AE_DDPM, self).validation_step(batch, **kwargs)

    def configure_optimizers(self, **kwargs):
        ae_params = self.ae_model.parameters()
        self.ddpm_optimizers = {
            arch_name: torch.optim.AdamW(
                params=self.all_target_models[arch_name][0].parameters(), lr=1e-3)
            for arch_name in self.all_target_models.keys()
        }
        self.ae_optimizer = torch.optim.AdamW(params=ae_params, lr=2e-4)
        return self.ddpm_optimizers, self.ae_optimizer