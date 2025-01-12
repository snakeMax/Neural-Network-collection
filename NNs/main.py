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
import tkinter.ttk as ttk
import os
import threading
from keras.layers import BatchNormalization
import pickle

# Create the Tkinter window
window = tk.Tk()
window.title("MNIST Digit Recognition")


# Create the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))


# Load the MNIST datasets
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Define the load_model_from_file function
def load_model_from_file():
    try:
        folder_path = filedialog.askdirectory()
        if folder_path:
            global model
            model = tf.keras.models.load_model(folder_path)
            model_name = os.path.basename(folder_path)
            model_name_label.config(text=f"Model loaded: {model_name}")
            model_name_entry.delete(0, tk.END)
            model_name_entry.insert(0, model_name)
            model_name_entry.config(state='readonly')
            try:
                parent_dir = os.path.dirname(folder_path)
                last_epoch_file = os.path.join(parent_dir, '_last_epoch.pkl')
                with open(last_epoch_file, 'rb') as f:
                    last_epoch = pickle.load(f)
                    epoch_label['text'] = f"Epochs: {last_epoch}/{last_epoch}"
            except FileNotFoundError:
                # If the file doesn't exist, set the epoch label to a default value
                epoch_label['text'] = f"Epochs: 0/0"
    except Exception as e:
        print(f"Error loading model: {e}")
        model_name_label.config(text="Error loading model")

def save_model():
    try:
        file_path = filedialog.asksaveasfilename()
        if file_path:
            model.save(file_path)
            with open(file_path + '_last_epoch.pkl', 'wb') as f:
                try:
                    last_epoch = epoch_label['text'].split('/')[0]
                    pickle.dump(int(last_epoch), f)
                except ValueError:
                    print("Error: Invalid epoch value")
    except Exception as e:
        print(f"Error saving model: {e}")


# Define the train_model function
def train_model():
    try:
        global model
        model_name = model_name_label['text'].split(': ')[1]
        if not model:
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(10, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        epochs = int(epochs_entry.get())
        epoch_label['text'] = f"Epochs: 0/{epochs}"  # Update the epoch label
        stop_training = threading.Event()
        def train_model_thread(model_name, stop_training):
            epoch_callback = EpochCallback(0, epochs)
            try:
                model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_data=(x_test, y_test), verbose=0, 
                          callbacks=[epoch_callback])
                with open(model_name + '_last_epoch.pkl', 'wb') as f:
                    pickle.dump(epoch_callback.epoch, f)
            except Exception as e:
                print(f"Error training model: {e}")
            finally:
                interrupt_button.pack_forget()  # Remove the interrupt button from the window
        interrupt_button = tk.Button(window, text="Interrupt Training", command=stop_training.set)
        interrupt_button.pack()
        thread = threading.Thread(target=train_model_thread, args=(model_name, stop_training))
        thread.start()
    except Exception as e:
        print(f"Error training model: {e}")

class EpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch, epochs):
        self.epoch = epoch
        self.epochs = epochs
        self.batch_count = 0
        self.batch_size = 128
        self.total_batches = len(x_train) // self.batch_size

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        progress = (self.batch_count / self.total_batches) * 100
        progress_bar['value'] = progress
        progress_bar.update_idletasks()  # Update the progress bar

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        epoch_label['text'] = f"Epochs: {self.epoch}/{self.epochs}"
        progress_bar['value'] = 0  # Reset the progress bar to 0
        progress_bar.update_idletasks()  # Update the progress bar
        self.batch_count = 0  # Reset the batch count


# Define the test_model function
def test_model():
    try:
        image_path = filedialog.askopenfilename()
        image = Image.open(image_path)
        image = image.resize((28, 28))  # Resize the image to 28x28
        image = image.convert('L')  # Convert the image to grayscale
        image = np.array(image)
        image = image.reshape((1, 28, 28, 1))  # Reshape the image to the correct dimensions
        image = image.astype('float32') / 255  # Normalize the image data
        prediction = model.predict(image)
        prediction_label.config(text="Prediction: " + str(np.argmax(prediction)))
        image_canvas.delete("all")
        resized_image = image[0, :, :, 0].astype('float32') * 255  # Normalize the image data to a range of 0 to 255
        resized_image = resized_image.astype(np.uint8)  # Convert the image to uint8
        resized_image = Image.fromarray(resized_image)  # Convert the image to a PIL image
        resized_image = resized_image.resize((280, 280))  # Resize the image to 280x280
        image_tk = ImageTk.PhotoImage(resized_image)
        image_canvas.create_image(0, 0, image=image_tk, anchor='nw')
        image_canvas.image = image_tk  # Keep a reference to the image
        print(f"Predicted digit: {np.argmax(prediction)}")
    except Exception as e:
        print(f"Error testing model: {e}")


# Create the Tkinter model name entry
model_name_label = tk.Label(window, text="Model Name:")
model_name_label.pack()
model_name_entry = tk.Entry(window)
model_name_entry.pack()

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

# Create the Tkinter epochs entry
epochs_label = tk.Label(window, text="Epochs:")
epochs_label.pack()
epochs_entry = tk.Entry(window)
epochs_entry.insert(0, "10")
epochs_entry.pack()

# Create the Tkinter progress bar
progress_bar = ttk.Progressbar(window, orient='horizontal', length=200, mode='determinate')
progress_bar.pack()

# Create the Tkinter epoch label
epoch_label = tk.Label(window, text="Epochs: 0")
epoch_label.pack()


# Start the Tkinter event loop
window.mainloop()