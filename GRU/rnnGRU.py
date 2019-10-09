import music21
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout, Activation
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import glob
from tqdm import tqdm
import numpy as np
import collections
import random
import matplotlib.pyplot as plt

def create_dict(notes):
    pitchnames = sorted(set(item for item in notes))
    dictionary = dict((note, number) for number, note in enumerate(pitchnames))
    return dictionary

def get_songs():
    fplist = glob.glob(r"C:\Users\seanf\dataset\*") #returns all file paths of the midi files, change this to your local dataset path
    notes = [] #notes in final form before dict
    notes_to_parse = [] #raw note objects
    for fp in tqdm(fplist): #for every file path with countdown
        score = music21.converter.parse(fp) #turn the midi into a score object
        part = music21.instrument.partitionByInstrument(score) #our midis have 1 piano object and so cant simply flatten
        notes_to_parse = part.parts[0].recurse() #get every note from the piano part
        for element in notes_to_parse:
            if isinstance(element, music21.note.Note): #if its a note
                notes.append(str(element.pitch)) #simply append the pitch
            elif isinstance(element, music21.chord.Chord): #if its a chord
                notes.append('.'.join(str(n) for n in element.normalOrder)) #append each chords notes, with a . between them indicating its part of a chord
            #TODO implement something that can parse rests, implement something that can keep track of velocity etc
    return notes
	
notes = get_songs()

def get_dict(notes): #converts the contents of a list to the dictionary equivalents
    dictionary = create_dict(notes) #get the dictionary
    notes_after_dic = [dictionary[note] for note in notes] #for every note in notes, look up the note and add to array
    notes_after_dic = np.array(notes_after_dic) #convert to a numpy array
    return notes_after_dic
	
sequence_length = 100
dictionary = create_dict(notes) #get the actual dictionary

print("Number of notes: ", len(notes))
print("Number of unique notes and chords: ", len(dictionary))
network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i+sequence_length]
    sequence_out = notes[i+sequence_length]
    network_input.append([dictionary[char] for char in sequence_in])
    network_output.append(dictionary[sequence_out])

n_patterns = len(network_input)
print("number of patterns: ", n_patterns)
n_vocab = len(dictionary)

#reshape the input into a format compatible with GRU layers, and one hot encode
X = np.reshape(network_input, (n_patterns, sequence_length, 1))
#normalise input
X = X / float(n_vocab)  
Y = to_categorical(network_output)  
print(X.shape)
print(Y.shape)

#define model
model = Sequential()
model.add(GRU(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2]))),
model.add(Dropout(0.3)),
model.add(GRU(128, return_sequences=False)),
model.add(Dense(128))
model.add(Dropout(0.3)),
model.add(Dense(Y.shape[1], activation='softmax')),
adam = optimizers.Adam(clipnorm=5.) #add gradient clipping to Adam
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#train the model
filepath = "GRUweights.hdf5"    
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
)   

callbacks_list = [checkpoint]     
history = model.fit(X, Y, epochs=200, batch_size=64, callbacks=callbacks_list, validation_split=0.33)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()