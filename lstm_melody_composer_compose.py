# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:08:38 2015

@author: Konstantin
"""

import data_utils_compose
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.recurrent import LSTM
import numpy as np
import glob
from os import listdir

#User Info
print()
print("User Information:")
print("This is a tool for composing melodies to given chord sequences with a LSTM Recurrent Neural Network that has already been trained.")
print("It has been created in Fall 2015 by Konstantin Lackner under the supervision of Thomas Volk and Prof. Diepold at the Chair of Data Processing at the Technical University of Munich (TUM).")


chord_dir = './testData/chords/'
composition_dir = './testData/melody_composition/'

print("Put the chord sequences in MIDI format in the directory: %s. Nothing else but the MIDI files should be in that directory!" %(chord_dir))
print("The composed melodies will be stored in MIDI format in the directory: %s" %(composition_dir))
print("Chord notes must be between C2 and B2.")

print("LSTM RNN Composer:")

chord_files = glob.glob("%s*.mid" %(chord_dir))

composition_files = []
for i in range(len(chord_files)):
    composition_files.append('%d' %(i+1))

mel_lowest_note = 60

print()
print("Using the following files as Test Chords:")
print(chord_files)
print()
print("Melodies will be saved to the files:")
print(composition_files)
print()

import sys

#print("Choose a resolution factor. (e.g. Resolution_Factor=24: 1/8 Resolution, 12: 1/16 Resolution, 6: 1/32 Resolution, etc...)")
resolution_factor = 12 #24: 1/8 Resolution, 12: 1/16 Resolution, 6: 1/32 Resolution

chord_lowest_note, chord_highest_note, chord_ticks = data_utils_compose.getNoteRangeAndTicks(chord_files, res_factor=resolution_factor)

chord_roll = data_utils_compose.fromMidiCreatePianoRoll(chord_files, chord_ticks, chord_lowest_note,
                                                        res_factor=resolution_factor)

double_chord_roll = data_utils_compose.doubleRoll(chord_roll)

test_data = data_utils_compose.createNetInputs(double_chord_roll, seq_length=chord_ticks)

print("Loading Model and Weights...")

#Load model file
model_dir = './weights/saved_model/'
model_files = list(filter(lambda x : x.endswith(".json"),listdir(model_dir)))

file_number_model = 0
model_file = model_files[file_number_model]
model_path = '%s%s' %(model_dir, model_file)

#Load weights file
weights_dir = './weights/'
weights_files = list(filter(lambda x : x.endswith(".h5"),listdir(weights_dir)))

file_number_weights = 0
weights_file = weights_files[file_number_weights]
weights_path = '%s%s' %(weights_dir, weights_file)

print("loading model...")
print(model_path)
model = model_from_json(open(model_path).read())

print("loading weights...")
print(weights_path)
model.load_weights(weights_path)

print("Compiling model...")
model.compile(loss='binary_crossentropy', optimizer='adam')

thresh = 0.1 # threshold is used for creating a Piano Roll Matrix out of the Network Output
print("Compose...")
for i, song in enumerate(test_data):
    net_output = model.predict(song)
    net_roll = data_utils_compose.NetOutToPianoRoll(net_output, threshold=thresh)
    data_utils_compose.createMidiFromPianoRoll(net_roll, mel_lowest_note, composition_dir, composition_files[i], thresh, res_factor=resolution_factor)    
    print("Finished composing song %d." %(i+1))

