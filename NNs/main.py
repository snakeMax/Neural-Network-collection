# Import necessary libraries
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# Create the Tkinter window
window = tk.Tk()
window.title("MNIST Digit Recognition")


# Create the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the input data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normalize the input data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Define the load_model_from_file function
def load_model_from_file():
    try:
        file_path = filedialog.askopenfilename()
        global model
        model = tf.keras.models.load_model(file_path)
    except Exception as e:
        print(f"Error loading model: {e}")

def save_model():
    try:
        file_path = filedialog.asksaveasfilename()
        model.save(file_path)
    except Exception as e:
        print(f"Error saving model: {e}")

# Define the train_model function
def train_model():
    try:
        for epoch in range(10):
            model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_test, y_test))
            training_progress_label.config(text=f"Epoch {epoch+1}/10")
            window.update()
        training_progress_label.config(text="Training complete!")
    except Exception as e:
        print(f"Error training model: {e}")

# Define the test_model function
def test_model():
    try:
        image_path = filedialog.askopenfilename()
        image = Image.open(image_path)
        image = image.resize((28, 28))
        image = np.array(image)
        image = image.reshape((1, 28, 28, 1))
        image = image.astype('float32') / 255
        prediction = model.predict(image)
        prediction_label.config(text=np.argmax(prediction))
        image_canvas.delete("all")
        image_tk = ImageTk.PhotoImage(Image.fromarray(image[0, :, :, 0]))
        image_canvas.create_image(0, 0, image=image_tk, anchor='nw')
        image_canvas.image = image_tk  # Keep a reference to the image
    except Exception as e:
        print(f"Error testing model: {e}")


# Create the Tkinter buttons
load_button = tk.Button(window, text="Load Model", command=load_model_from_file)
train_button = tk.Button(window, text="Train Model", command=train_model)
test_button = tk.Button(window, text="Test Model", command=test_model)
save_button = tk.Button(window, text="Save Model", command=save_model)

# Pack the buttons into the window
load_button.pack()
train_button.pack()
test_button.pack()
save_button.pack()

# Create the Tkinter image canvas
image_canvas = tk.Canvas(window, width=280, height=280)
image_canvas.pack()

# Create the Tkinter prediction label
prediction_label = tk.Label(window, text="")
prediction_label.pack()

# Create the Tkinter training progress label
training_progress_label = tk.Label(window, text="")
training_progress_label.pack()

# Start the Tkinter event loop
window.mainloop()