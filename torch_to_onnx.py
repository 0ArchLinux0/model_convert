#!/usr/bin/python
# 필요한 import문
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch
from torch.utils.data import DataLoader
from datasets.landmark import Landmark
from utils.wing_loss import WingLoss
from models.slim import Slim
import sys
import time
from utils.consoler import rewrite, next_line

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('device to run:',device)

model = Slim()
model.load_state_dict(torch.load('./pretrained_weights/slim_160_latest.pth', map_location=torch.device('cpu')))
model.eval()

inputs = torch.randn(16, 3, 3, 3).to(device)
# 모델 변환
torch.onnx.export(model,               # 실행될 모델
                  inputs,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "./slim_160_latest.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

print("convert!")
