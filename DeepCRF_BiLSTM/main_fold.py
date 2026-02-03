import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from TorchCRF  import CRF  # Conditional Random Fields
import train as trRelu
import os
import sys  # Import sys for exiting the program
import tensorflow as tf


import train_DeepCRF_Fold as dcrf_fold
if __name__ == "__main__":  # Ensure this runs only when executing main.py

    # need ravdess.feat Ref: train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\ravdess.feat"
    # change path of her: train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\" + dataset + ".feat"
    # at def train_DeepCRF(odataset, dataset_name): funtion

    # will save graph image at drawing.py
    # path_loss = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\" + dataset_name + "_Training_loss_vs_Validation_loss.png"
    # path_acc = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\" + dataset_name + "_Training_accuracy_vs_Validation_accuracy.png"

    # cremaD # emodb # savee # tess # r_t_s_c_e                       # r_t_s_e                  # r_t_c #
    # CremaD # EmoDB # Savee # Tess # Ravdess_Tess_Savee_CremaD_EmoDB # Ravdess_Tess_Savee_EmoDB # Ravdess_Tess_CremaD
    dataset = "ravdess"
    dataset_name= "Ravdess"
    dcrf_fold.train_DeepCRF(dataset, dataset_name)
