import tensorflow as tf
import os

# Load data
data = []
labels = []
for label in ["0", "1"]:
    directory = f"recordings/{label}"
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            path = os.path.join(directory, filename)
            data.append(tf.audio.decode_wav(tf.io.read_file(path))[0].numpy())
            labels.append(int(label))

# Convert data and labels to NumPy arrays
data = tf.keras.preprocessing.sequence.pad_sequences(data, dtype="float32")
labels = tf.keras.utils.to_categorical(labels)

# Split data into training and testing sets
split_index = int(0.8 * len(data))
x_train, y_train = data[:split_index], labels[:split_index]
x_test, y_test = data[split_index:], labels[split_index:]

# Create neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(data.shape[1], data.shape[2])),
    tf.keras.layers.Conv1D(filters=8, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation="softmax")
])

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")