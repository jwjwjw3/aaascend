import numpy as np
import cv2
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path
import vai_q_onnx
import argparse

import torch
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod, quantize_static

import os, shutil
from time import perf_counter

def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default ='ResNet10_1111_8.U8S8.onnx')
args = parser.parse_args()

datafile = r'./data/cifar-10-batches-py/test_batch'
metafile = r'./data/cifar-10-batches-py/batches.meta'

data_batch_1 = unpickle(datafile) 
metadata = unpickle(metafile)

images = data_batch_1['data']
labels = data_batch_1['labels']
images = np.reshape(images,(10000, 3, 32, 32))

dirname = 'images'
if not os.path.exists(dirname):
   os.mkdir(dirname)

quant_dir = './models/quant'

ipu_time = []

quantized_model_path = quant_dir + '/' + args.model
print(quantized_model_path.encode('unicode_escape'))
model = onnx.load(quantized_model_path.encode('unicode_escape'))

providers = ['CPUExecutionProvider']
provider_options = [{}]

session = ort.InferenceSession(model.SerializeToString(), providers=providers,
                            provider_options=provider_options)

elapsed = 0
correct = 0
for i in range(10000): 
    im = images[i]
    image_array = np.array(im).astype(np.float32)
    image_array = image_array/255
    input_data = np.expand_dims(image_array, axis=0)
    
    ts = perf_counter()
    outputs = session.run(None, {'input': input_data})
    elapsed += perf_counter() - ts

    # output_array = outputs[0]
    # predicted_class = np.argmax(output_array)
    # predicted_label = metadata['label_names'][predicted_class]
    # label = metadata['label_names'][labels[i]]
    # if predicted_class == labels[i]:
    #     correct += 1
ipu_time.append(elapsed)
# summary(model, (3, 32, 32))
print(f"Model: {args.model}, Accuracy: {correct/10000}, Time: {elapsed}")
    