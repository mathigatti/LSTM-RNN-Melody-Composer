# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:08:18 2015

@author: Konstantin
"""

import data_utils_train
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import numpy as np
import time
import glob

#User Info
print()
print("User Information:")
print("This is a tool for training a LSTM Recurrent Neural Network to learn melodies to given chord sequences.")
print("It has been created in Fall 2015 by Konstantin Lackner under the supervision of Thomas Volk and Prof. Diepold at the Chair of Data Processing at the Technical University of Munich (TUM).")

chord_train_dir = './trainData/chords/'
mel_train_dir = './trainData/melody/'

print("Put the chords in MIDI format in the directory: %s" %(chord_train_dir))
print("Put the melodies in MIDI format in the directory: %s" %(mel_train_dir))
print("ALL MIDI files need to be of the same length (e.g. 8 bars).")
print("Chord notes must be between C2 and B2. Melody notes must be between C3 and B4.")
print("Keep the chord and CORRESPONDING melody files in the SAME ORDER within their respective folders.")
print()
print()
print("LSTM RNN Trainer:")
print()
chord_train_files = glob.glob("%s*.mid" %(chord_train_dir))
mel_train_files = glob.glob("%s*.mid" %(mel_train_dir))

import sys

#resolution_factor = int(input('Resolution Factor (recommended=12):')) #24: 1/8 Resolution, 12: 1/16 Resolution, 6: 1/32 Resolution
resolution_factor = 12 #24: 1/8 Resolution, 12: 1/16 Resolution, 6: 1/32 Resolution

#Preprocessing: Get highest and lowest notes + maximum midi_ticks overall midi files
chord_lowest_note, chord_highest_note, chord_ticks = data_utils_train.getNoteRangeAndTicks(chord_train_files, res_factor=resolution_factor)
mel_lowest_note, mel_highest_note, mel_ticks = data_utils_train.getNoteRangeAndTicks(mel_train_files, res_factor=resolution_factor)

#Create Piano Roll Representation of the MIDI files. Return: 3-dimensional array or shape (num_midi_files, maximum num of ticks, note range)
chord_roll = data_utils_train.fromMidiCreatePianoRoll(chord_train_files, chord_ticks, chord_lowest_note, chord_highest_note,
                                                res_factor=resolution_factor)
mel_roll = data_utils_train.fromMidiCreatePianoRoll(mel_train_files, mel_ticks, mel_lowest_note, mel_highest_note,
                                              res_factor=resolution_factor)

#Double each chord_roll and mel_roll. Preprocessing to create Input and Target Vector for Network
double_chord_roll = data_utils_train.doubleRoll(chord_roll)
double_mel_roll = data_utils_train.doubleRoll(mel_roll)

#Create Network Inputs:
#Input_data Shape: (num of training samples, num of timesteps=sequence length, note range)
#Target_data Shape: (num of training samples, note range)
input_data, target_data = data_utils_train.createNetInputs(double_chord_roll, double_mel_roll, seq_length=chord_ticks)
input_data = input_data.astype(np.bool)
target_data = target_data.astype(np.bool)


input_dim = (input_data.shape[1], input_data.shape[2])
output_dim = target_data.shape[1]

num_epochs = int(sys.argv[1])

batch_size = 128

print("Network Input Dimension:", input_dim)
print("Network Output Dimension:", output_dim)

num_layers = 3
num_units = [6,12]

#Building the Network
model = Sequential()
if num_layers == 1:
    model.add(LSTM(input_shape=input_dim, output_shape=output_dim, activation='sigmoid', return_sequences=False))
elif num_layers > 1:
    model.add(LSTM(num_units[0],input_shape=input_dim, activation='sigmoid', return_sequences=True))
    for i in range(num_layers-2):
        model.add(LSTM(num_units[i+1], activation='sigmoid', return_sequences=True))
    model.add(LSTM(output_dim=output_dim, activation='sigmoid', return_sequences=False))


print("Compiling your network with the following properties:")
loss_function = 'binary_crossentropy'
optimizer = 'adam'

print("Loss function: ", loss_function)
print("Optimizer: ", optimizer)
print("Number of Epochs: ", num_epochs)
print("Batch Size: ", batch_size)

model.compile(loss=loss_function, optimizer=optimizer)


print("Training...")

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    

checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
)  

model.fit(input_data, target_data, batch_size=batch_size, epochs=num_epochs, callbacks=[checkpoint])

print()
print("Saving model and weights...")
print("Saving weights...")
weights_dir = './weights/'
weights_file = '%dlayer_%sepochs_%s' %(num_layers, num_epochs, time.strftime("%Y%m%d_%H_%M.h5"))
weights_path = '%s%s' %(weights_dir, weights_file)
print("Weights Path:", weights_path)
model.save_weights(weights_path)

print("Saving model...")
json_string = model.to_json()
model_file = '%dlayer_%sepochs_%s' %(num_layers, num_epochs, time.strftime("%Y%m%d_%H_%M.json"))
model_dir = './saved_model/'
model_path = '%s%s' %(model_dir, model_file)
print("Model Path:", model_path)
open(model_path, 'w').write(json_string)