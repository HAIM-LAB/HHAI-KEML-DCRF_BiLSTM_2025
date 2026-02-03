import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from TorchCRF  import CRF  # Conditional Random Fields
import train as trRelu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode
os.environ["OMP_NUM_THREADS"] = "50"  # Adjust based on CPU cores
os.environ["TF_NUM_INTEROP_THREADS"] = "35"
os.environ["TF_NUM_INTRAOP_THREADS"] = "35"
import sys  # Import sys for exiting the program
import tensorflow as tf
import time
# from tensorflow.keras.layers import Bidirectional
# print(dir(tf.keras.layers))

# dataset = "ravdess"
# dataset_name= "Ravdess"

# dataset = "savee"
# dataset_name= "Savee"

#dataset = "tess"
#dataset_name= "Tess"

import train_DeepCRF_PCA_Count as dcrf
#import train_lstm_auc as tdcrf
if __name__ == "__main__":  # Ensure this runs only when executing main.py

    # need ravdess.feat Ref: train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\ravdess.feat"
    # change path of her: train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\" + dataset + ".feat"
    # at def train_DeepCRF(dataset, dataset_name): funtion

    # will save graph image at drawing.py
    # path_loss = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\" + dataset_name + "_Training_loss_vs_Validation_loss.png"
    # path_acc = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\" + dataset_name + "_Training_accuracy_vs_Validation_accuracy.png"

    dataset = "ravdess"
    dataset_name= "Ravdess"
    dcrf.train_DeepCRF(dataset, dataset_name)
    #sys.exit(0)
    
    # need tess.feat Ref: train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\tess.feat"
    # change path of her: train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\" + dataset + ".feat"
    # at def train_DeepCRF(dataset, dataset_name): funtion
    dataset = "tess"
    dataset_name= "Tess"
    dcrf.train_DeepCRF(dataset, dataset_name)
    #sys.exit(0)

    # need savee.feat Ref: train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\savee.feat"
    # change path of her: train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\" + dataset + ".feat"
    # at def train_DeepCRF(dataset, dataset_name): funtion
    dataset = "savee"
    dataset_name= "Savee"
    dcrf.train_DeepCRF(dataset, dataset_name)
    #sys.exit(0)

    dataset = "emodb"
    dataset_name= "EmoDB"
    dcrf.train_DeepCRF(dataset, dataset_name)
    #sys.exit(0)

    dataset = "cremaD"
    dataset_name= "CremaD"
    dcrf.train_DeepCRF(dataset, dataset_name)
    #sys.exit(0)

    #dataset = "r_t_e"
    #dataset_name= "Ravdess_Tess_EmoDB"
    #dcrf.train_DeepCRF(dataset, dataset_name)
    #print("Start for sleeping")
    #time.sleep(60*3)  # Pause execution for 2 seconds
    #print("Wakeup....")
    dataset = "r_t_s"
    dataset_name= "Ravdess_Tess_Savee"
    dcrf.train_DeepCRF(dataset, dataset_name)

    dataset = "r_t_s_c_e"
    dataset_name= "Ravdess_Tess_Savee_CremaD_EmoDB"
    dcrf.train_DeepCRF(dataset, dataset_name)

    #dataset = "r_t_s_e"
    #dataset_name= "Ravdess_Tess_Savee_EmoDB"
    #dcrf.train_DeepCRF(dataset, dataset_name)
 

    #dataset = "r_t_c"
    #dataset_name= "Ravdess_Tess_CremaD"
    #dcrf.train_DeepCRF(dataset, dataset_name)

    # need r_t_s_c_e.feat Ref: train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\r_t_s_c_e.feat"
    # change path of her: train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\" + dataset + ".feat"
    # at def train_DeepCRF(dataset, dataset_name): funtion
