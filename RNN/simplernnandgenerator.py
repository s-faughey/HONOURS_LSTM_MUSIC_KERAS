import music21
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Dense, Dropout, Activation
from keras.utils import to_categorical
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
    fplist = glob.glob(r"C:\Users\seanf\dataset\*") #returns all file paths of the midi files
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
    return notes

notes = get_songs()
print(notes)

def get_dict(notes): #converts the contents of a list to the dictionary equivalents
    dictionary = create_dict(notes) #get the dictionary
    notes_after_dic = [dictionary[note] for note in notes] #for every note in notes, look up the note and add to array
    notes_after_dic = np.array(notes_after_dic) #convert to a numpy array
    return notes_after_dic

sequence_length = 100
dictionary = create_dict(notes) #get the actual dictionary

#summarise the loaded data
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

#reshape the input into a format compatible with SimpleRNN layers, and one hot encode
X = np.reshape(network_input, (n_patterns, sequence_length, 1))
#normalise input
X = X / float(n_vocab)  
Y = to_categorical(network_output)  
print(X.shape)
print(Y.shape)

#define model
model = Sequential()
model.add(SimpleRNN(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2]))),
model.add(Dropout(0.3)),
model.add(SimpleRNN(128, return_sequences=False)),
model.add(Dense(128)),
model.add(Dropout(0.3)),
model.add(Dense(Y.shape[1], activation='softmax')),
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the model
filepath = "SimpleRNNweights"    
filepathhdf5 = filepath+".hdf5"
checkpoint = ModelCheckpoint(
    filepathhdf5, monitor='loss', 
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

#generate random starting note
starting_note = random.randint(0,len(X)-1)
pattern = X[starting_note]
output = []
#generate random length between 300-800 (roughly between 1 and 5 minutes of music)
song_length = random.randint(300, 800)
print(len(pattern))

pitchnames = sorted(set(item for item in notes))
int_to_note = dict((i, c) for i, c in enumerate(pitchnames))
print(int_to_note)


#generate predictions
for i in range(song_length):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)
    
    #extract index of highest value
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    #collect all outputs from the network
    result = int_to_note[index]
    output.append(result)
    
    pattern = np.append(pattern, index)
    pattern = pattern[1:len(pattern)]
print(output)

#output music
offset = 0
output_notes = []
# create note and chord objects based on the values generated by the model
for pattern in output:
    # pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = music21.note.Note(int(current_note))
            new_note.storedInstrument = music21.instrument.Piano()
            notes.append(new_note)
        new_chord = music21.chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # pattern is a note
    else:
        new_note = music21.note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = music21.instrument.Piano()
        output_notes.append(new_note)
    # increase offset each iteration so that notes do not stack
    offset += 0.5

midi_stream = music21.stream.Stream(output_notes)
midi_stream.write('midi', fp=filepath+'.mid')
