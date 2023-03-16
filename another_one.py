import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# set the directory path for the recordings
dir_path = './recordings'

# create a list of file names in the directory
files = os.listdir(dir_path)

# load the audio data and labels from the files
data = []
labels = []
for file in files:
    label = int(file[0])
    audio_path = os.path.join(dir_path, file)
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    audio_data = audio_data.reshape(1, audio_data.shape[0])
    data.append(audio_data)
    labels.append(label)

# convert the audio data and labels to numpy arrays
data = np.vstack(data)
labels = np.array(labels)

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(1, 8000, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# evaluate the model on the testing set
loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)