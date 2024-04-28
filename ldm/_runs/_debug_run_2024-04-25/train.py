import sys
import torch

sys.path.append("../..")

from training.ddpm.ae_ddpm import AE_DDPM
from diffResFormer.encoder import small
from diffResFormer.unet import AE_CNN_bottleneck

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
    trainer = AE_DDPM(
                ae_model=small(in_dim=512, input_noise_factor=0.1, latent_noise_factor=0.1), 
                model=AE_CNN_bottleneck(in_dim=512, in_channel=1, time_step=50, dec=None),
                config_dict=config_dict)

    print(trainer.generate(batch=torch.rand(10, 1, 512).cuda()).shape)


if __name__ == "__main__":
    train()