import pandas as pd
import numpy as np
import random
import math
import argparse
import time
import torch.nn.functional as F
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import sklearn.metrics as metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC,LinearSVC
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x=torch.tensor(3.0,requires_grad=True)
y=torch.pow(x,2)
print(x,y)

#判断x,y是否是可以求导的
print(x.requires_grad)
print(y.requires_grad)

#求导，通过backward函数来实现
y.backward()

#查看导数，也即所谓的梯度
print(x.grad)
print(y.grad)

