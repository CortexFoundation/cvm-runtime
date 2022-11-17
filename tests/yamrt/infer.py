import onnxruntime
import numpy as np
import cv2
import json
import os
import time

providers = ['CPUExecutionProvider']
session = onnxruntime.InferenceSession('resnet18v1/resnet18v1.onnx', providers=providers)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

labels = load_labels('labels.json')

path = '/home/remloveh/.mxnet/datasets/imagenet/val/'
li = os.listdir(path)
li.sort()

total = 0
right = 0
label_num = 0
start = time.time()
for di in li:
    ll = os.listdir(path + di)
    for dd in ll:
        img = cv2.imread(path + di + '/' + dd)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        data = np.array(img).transpose(2, 0, 1)
        data = data.astype('float32')
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_data = np.zeros(data.shape).astype('float32')
        
        for i in range(data.shape[0]):
            norm_data[i,:,:] = (data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        norm_data = norm_data.reshape(1, 3, 224, 224).astype('float32')

        result = session.run([output_name],{input_name:norm_data})
        res = softmax(np.array(result)).tolist()
        idx = np.argmax(res)
        
        if idx == label_num:
            right = right + 1
        total = total + 1
    label_num = label_num + 1
end = time.time()

print('time: ', int(end - start)/ total, 's')


print('accuracy: ', right/total)
