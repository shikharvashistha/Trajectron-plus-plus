# So I've been thinking to make a model which I'm going to call 
# adversagaurd which will take input from nucense data set and 
# use attention mechanism for finding the appropriate features that affects 
# ADE(average displacement error), FDE and the metrics which basically affects the 
# trajectory upon adversarial black box or white box attacks
# and upon extracting(attention) the features/inputs will be fed to a DRL model
# which will inturn give me policy and value function which will be used to
# minimize the ADE and FDE and the metrics which affects the trajectory upon adversarial attacks
# upon getting probability distribution from the DRL model I will use the lower bound and upper bound as a range
# if the output of a model falls in the range then it will be considered as a valid output otherwise it will be considered as an invalid output
# and this invalid output will be used to train the model to minimize the ADE and FDE and the metrics which affects the trajectory upon adversarial attacks
# the input to the DRL model will be the output of the attention mechanism and a error rate which will signify the amount of error that can be tolerated
# help me code this model in python and I'll name this model as adversagaurd
# what should I use pytorch, tensorflow or keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import time
import sys
import pickle
import argparse
import logging
import datetime
import glob
import json
import shutil


class Adversagaurd(object):
    def __init__(self, registrar, hyperparams, log_writer, device):
        self.registrar = registrar
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0
        self.node_models_dict = {}
        self.edge_models_dict = {}
        self.edge_types = []
        self.ph = None
        self.log_writer = log_writer
        self.log_writer.write('Adversagaurd initialized.')
    