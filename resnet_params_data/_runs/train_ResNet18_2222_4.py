import sys, os

sys.path.append("../utils")
from utils import *

model_name = "ResNet18_2222_4"
all_models_folder = "../models"
dataset_folder = "../cifar10_dataset"

for seed in range(200, 500):
    cur_model_folder = os.path.join(all_models_folder, model_name)
    cur_log_filename = "trainlog_"+model_name+"_seed_"+str(seed)+".torch"
    if not os.path.isfile(os.path.join(cur_model_folder, cur_log_filename)):
        resnet_train(model_name=model_name, random_seed=seed, dataset_folder=dataset_folder, 
            model_folder=os.path.join(all_models_folder, model_name))