# ONE LIBRARY REQUIRED FOR RUNNING THIS CODE! (aside from torch)
# !pip install bi_lstm_crf

import torch
import pandas as pd
import numpy as np
import re
import os
import time  # For eta
from pathlib import Path

# For data management
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Torch imports
import torch.nn as nn
import torch.optim as optim
from bi_lstm_crf import CRF
import torch.nn.functional as F

OUT_CHANNELS = 40
CONV_KERNEL_SIZE = 3
POOL_KERNEL_SIZE = 3
LSTM_UNITS = 128
DROPOUT_EM = 0.3
WINDOW_SIZE = 5
ALPHABET_SIZE = 30
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
MAX_TOKEN_LENGTH = 30

LSTM_IN = ((WINDOW_SIZE+2)-POOL_KERNEL_SIZE+1) * OUT_CHANNELS

# For preprocessing word
# For padding / converting all input to np array
def convert(input_string, input_type, direction='code'):
    if direction=='code':
        if input_type == 'word':
            return [(ALPHABET.index(i)+1) for i in input_string] + [0]*(MAX_TOKEN_LENGTH-len(input_string))
        elif input_type == 'solution':
            return [int(i) for i in input_string] + [0]*(MAX_TOKEN_LENGTH-len(input_string))
    if direction=='decode':
        if input_type == 'word':
            return ''.join([ALPHABET[i-1] for i in input_string if i!=0])
        elif input_type == 'solution':
            return ''.join([str(i) for i in input_string if i!=0])

# Shorthand for converter
encode_x = lambda input_string: convert(input_string, input_type='word', direction='code')
encode_y = lambda input_string: convert(input_string, input_type='solution', direction='code')
decode_x = lambda input_string: convert(input_string, input_type='word', direction='decode')
decode_y = lambda input_string: convert(input_string, input_type='solution', direction='decode')

# For creating windowed version of input
def expand(x_input):
    chars = len(ALPHABET)
    window_pad = WINDOW_SIZE//2
    window = list(range(chars, chars+window_pad*2))
    word_padded = np.concatenate([window[0:window_pad], x_input[x_input!=0], window[window_pad:]])
    
    full_list = []
    start_index=0
    for i, char in enumerate(x_input):
        if char==0:
            full_list.append([0]*WINDOW_SIZE)
        else:
            full_list.append(word_padded[start_index:start_index+WINDOW_SIZE])
            start_index+= 1
            
    full_list = np.array(full_list)  
    return full_list  

# Pytorch modeling
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Define model
class syl_model(nn.Module):

    def __init__(self):
        super(syl_model, self).__init__()
        self.em = nn.Embedding(ALPHABET_SIZE, 128)
        self.dropout1 = nn.Dropout(p=DROPOUT_EM)
        self.conv1 =  torch.nn.Conv1d(in_channels=128, out_channels=OUT_CHANNELS, kernel_size=CONV_KERNEL_SIZE, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(OUT_CHANNELS)  # Add batch normalization after conv layer
        self.pool = torch.nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=1, padding=1)
        
        self.lstm = nn.LSTM(LSTM_IN, LSTM_UNITS, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(LSTM_UNITS*2)  # Add batch normalization after LSTM
        self.crf = CRF(LSTM_UNITS*2, 3)

    def forward(self, x):

        batch_size=x.size(0)
        mask = x[:,:,2].gt(0)
        
        x = x.reshape(batch_size*MAX_TOKEN_LENGTH, WINDOW_SIZE)
        x = self.em(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = x.reshape(batch_size, MAX_TOKEN_LENGTH, LSTM_IN)      
        x, _ = self.lstm(x)

        x = x.permute(0, 2, 1)  # Change shape for batch norm
        x = self.bn2(x)  # Apply batch norm after LSTM
        x = x.permute(0, 2, 1)  # Change shape back  
        
        scores, tag_seq = self.crf(x, mask)      
        return scores, tag_seq

    def loss(self, x, tags):

        batch_size=x.size(0)
        mask = x[:,:,2].gt(0)
        
        x = x.reshape(batch_size*MAX_TOKEN_LENGTH, WINDOW_SIZE)
        x = self.em(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)      
        x = F.relu(x)
        x = self.pool(x)             
        x = torch.flatten(x, start_dim=1)              
        x = x.reshape(batch_size, MAX_TOKEN_LENGTH, LSTM_IN)                 
        x, _ = self.lstm(x)

        x = x.permute(0, 2, 1)  # Change shape for batch norm
        x = self.bn2(x)  # Apply batch norm after LSTM
        x = x.permute(0, 2, 1)  # Change shape back  
        
        loss = self.crf.loss(x, tags, mask)
        return loss

# Create instance, load weights
neural_model = syl_model().to(device)
file_path = Path(__file__).resolve().parent
model_path = str(file_path)+'\\weights.pt'
neural_model.load_state_dict(torch.load(model_path, weights_only=True))

# Simple function for using model to syllabify
def torch_syllabificate(word):

    word_list = word
    target_words = word_list
    target_words = pd.Series(target_words.split())
    target_words = np.array(target_words.apply(encode_x).to_list())
    target_words_windowed = np.empty((len(target_words), MAX_TOKEN_LENGTH, WINDOW_SIZE),dtype=np.int32)
    for i in range(len(target_words)):
        target_words_windowed[i] = expand(target_words[i])
    target_words_windowed = torch.tensor(target_words_windowed, dtype=torch.long)

    # Run words through model
    neural_model.eval()
    target_outputs = neural_model(target_words_windowed)
    target_outputs = target_outputs[1]
    number_words = len(target_outputs)
    target_outputs = torch.nested.nested_tensor(target_outputs)
    target_outputs = torch.nested.to_padded_tensor(target_outputs, padding=0,output_size=(number_words,MAX_TOKEN_LENGTH))

    # Output as list of strings
    output = []
    for word in enumerate(word_list.split()):
        current_word = []
        for char in enumerate(word[1]):
            if target_outputs[word[0]][char[0]]==1:
                current_word += [char[1]]
            else:
                current_word += [char[1] + '-']
        current_word = ''.join(current_word)
        output = output + [current_word]

    #return output


def torch_syllabificate(word):

    word_list = word
    target_words = word_list
    target_words = pd.Series(target_words.split())
    target_words = np.array(target_words.apply(encode_x).to_list())
    target_words_windowed = np.empty((len(target_words), MAX_TOKEN_LENGTH, WINDOW_SIZE),dtype=np.int32)
    for i in range(len(target_words)):
        target_words_windowed[i] = expand(target_words[i])
    target_words_windowed = torch.tensor(target_words_windowed, dtype=torch.long)

    # Run words through model
    neural_model.eval()
    target_outputs = neural_model(target_words_windowed)
    target_outputs = target_outputs[1]
    number_words = len(target_outputs)
    target_outputs = torch.nested.nested_tensor(target_outputs)
    target_outputs = torch.nested.to_padded_tensor(target_outputs, padding=0,output_size=(number_words,MAX_TOKEN_LENGTH))

    # Output as list of strings
    output = []
    for word in enumerate(word_list.split()):
        current_word = []
        for char in enumerate(word[1]):
            if target_outputs[word[0]][char[0]]==1:
                current_word += [char[1]]
            else:
                current_word += [char[1] + '-']
        current_word = ''.join(current_word)
        output = output + [current_word]

    return output

        
