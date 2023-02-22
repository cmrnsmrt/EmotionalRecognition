import numpy as np # Used for multidimensional arrays
import seaborn as sns # Used for data visualisation
import matplotlib.pyplot as plt # For plotting the graph outputs
import utils
import os
import tensorflow as tf  # Using Tensorflow enables machine learning and artificial intelligence

# Keras model is used as a wrapper for the Tensorflow library
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json # Loads in model for making predictions
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D

from IPython.display import SVG, Image # Used to creat output images
from livelossplot.inputs.tf_keras import PlotLossesCallback

print("Tensorflow:", tf.__version__)

for expression in os.listdir("train/"): # Filepath for training data
    print("There were " + str(len(os.listdir("train/" + expression))) + " images found for the expression: " + expression)

# Control variables
img_size = 48 # Size of images in data set, represents width and height
batch_size = 64
epochs = 25 # Amounts of times the entire dataset is passed forwards and backwards through the neural network

dataGeneratorTrain = ImageDataGenerator(horizontal_flip=True) # Flips images horizontally
trainingGenerator = dataGeneratorTrain.flow_from_directory("train/", # Image directory for training
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)
dataGeneratorValidate = ImageDataGenerator(horizontal_flip=True) # Flips images horizontally
validationGenerator = dataGeneratorValidate.flow_from_directory("test/", # Image directory for testing
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

# Initialising the CNN
model = Sequential() # Setting up the model
'''

# Comment start here for load only

# 1st Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1))) # .add(Conv2D()) is used to specify that a convolutional layer is to be added to model
model.add(BatchNormalization()) # Applies a transformation that normalises model that keeps the mean close to 0 and the standard deviation close to 1
model.add(Activation('relu')) # Applies rectified linear unit transformation to model to perform non linear transformation
model.add(MaxPooling2D(pool_size=(2, 2))) # Takes only the maximum data from the previous step with pool 2 x 2
model.add(Dropout(0.25)) # Removes model excess and stops it overfitting. 0.25 mean 1/4 units are dropped

# 2nd Convolution - Kernal size is doubled and strides goes from (3,3) to (5,5)
model.add(Conv2D(128,(5,5), padding='same')) # .add(Conv2D()) is used to specify that a convolutional layer is to be added to model. Input shape was specified earlier.
model.add(BatchNormalization()) # Applies a transformation that normalises model that keeps the mean close to 0 and the standard deviation close to 1
model.add(Activation('relu')) # Applies rectified linear unit transformation to model to perform non linear transformation
model.add(MaxPooling2D(pool_size=(2, 2))) # Takes only the maximum data from the previous step with pool 2 x 2
model.add(Dropout(0.25)) # Removes model excess and stops it overfitting. 0.25 mean 1/4 units are dropped

# 3rd Convolution - Kernal size is quadrupled and strides goes from (5,5) to (3,3)
model.add(Conv2D(512,(3,3), padding='same')) # .add(Conv2D()) is used to specify that a convolutional layer is to be added to model. Input shape was specified earlier
model.add(BatchNormalization()) # Applies a transformation that normalises model that keeps the mean close to 0 and the standard deviation close to 1
model.add(Activation('relu')) # Applies rectified linear unit transformation to model to perform non linear transformation
model.add(MaxPooling2D(pool_size=(2, 2))) # Takes only the maximum data from the previous step with pool 2 x 2
model.add(Dropout(0.25)) # Removes model excess and stops it overfitting. 0.25 mean 1/4 units are dropped

# 4th Convolution layer 
model.add(Conv2D(512,(3,3), padding='same')) # .add(Conv2D()) is used to specify that a convolutional layer is to be added to model. Input shape was specified earlier
model.add(BatchNormalization()) # Applies a transformation that normalises model that keeps the mean close to 0 and the standard deviation close to 1
model.add(Activation('relu')) # Applies rectified linear unit transformation to model to perform non linear transformation
model.add(MaxPooling2D(pool_size=(2, 2))) # Takes only the maximum data from the previous step with pool 2 x 2
model.add(Dropout(0.25)) # Removes model excess and stops it overfitting. 0.25 mean 1/4 units are dropped

# Flattening
model.add(Flatten()) # Converts data into 1 dimensional array creating long feature vector

# 1st Fully connected layer
model.add(Dense(256)) # Adding dense layer which is a regular deeply connected neural network layer
model.add(BatchNormalization()) # Applies a transformation that normalises model that keeps the mean close to 0 and the standard deviation close to 1
model.add(Activation('relu'))  # Applies rectified linear unit transformation to model to perform non linear transformation
model.add(Dropout(0.25)) # Removes model excess and stops it overfitting

# 2nd Fully connected layer
model.add(Dense(512)) # Adding dense layer which is a regular deeply connected neural network layer
model.add(BatchNormalization()) # Applies a transformation that normalises model that keeps the mean close to 0 and the standard deviation close to 1
model.add(Activation('relu')) # Applies rectified linear unit transformation to model to perform non linear transformation
model.add(Dropout(0.25)) # Removes model excess and stops it overfitting. 0.25 mean 1/4 units are dropped
model.add(Dense(7, activation='softmax')) # Converts vector of values to a probability distribution that all sum up to equal one

# Compile and print summary
opt = Adam(lr=0.0005) # Stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) # Configures and creates the model so it can be used

model.summary() # Outputs summary of the model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True) # Prints out a png of model structure
Image('model.png',width=400, height=200) # Displays image

stepsPerEpoch = trainingGenerator.n//trainingGenerator.batch_size # Splits each epoch into a managable chunks
validation_steps = validationGenerator.n//validationGenerator.batch_size

reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001, mode='auto')  # Reduces the learning rate when a dop in performance is detected

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)

callbacks = [PlotLossesCallback(), checkpoint, reduction] # Gives live feedback on how the training is going by outputting graph

history = model.fit( # Starts training a validation of model
    x=trainingGenerator, # Training dataset declared earlier as 'train/'
    steps_per_epoch=stepsPerEpoch, # Calculated earlier by dividing number of images by batch size
    epochs=epochs, # Number of full passes back and forth
    validation_data = validationGenerator, ## Validaton dataset declared earlier as 'test/'
    validation_steps = validation_steps, # Calculated earlier by dividing numberof images by batch size
    callbacks=callbacks
) 

model_json = model.to_json() # Model is converted to JSON so it can used to make predictons
model.save_weights('model_weights.h5') # Weights for model are saved to different file

with open("model.json", "w") as JsonFile:
    JsonFile.write(model_json)

# Comment end here for load only
'''
class FacialExpressions(object):
    emotions = ["Anger", "Disgust",
                    "Fear", "Happiness",
                    "Neutrality", "Sadness",
                    "Surprise"]
    def __init__(self, modelJsonFile, modelWeightsFile):
        # load model from JSON file
        with open(modelJsonFile, "r") as JsonFile:
            loadedModelJson = JsonFile.read()
            self.loadedModel = model_from_json(loadedModelJson)
        # load weights into the new model
        self.loadedModel.load_weights(modelWeightsFile)
        self.loadedModel.make_predict_function()
    # Predict emotions from a given image
    def predictEmotion(self, img):
        self.predictions = self.loadedModel.predict(img)
        return FacialExpressions.emotions[np.argmax(self.predictions)]

import cv2
# OpenCV2 is used to apply the model to the problem
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressions("model.json", "model_weights.h5") # Loads in model and weights
font = cv2.FONT_HERSHEY_SIMPLEX # Sets font

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture('sample4.mp4') # Loads in video capture frame, change to file to predict on existing video
    def __del__(self):
        self.video.release() # Deletes loaded in video capture frame
    def getFrame(self): # Creates predictions
        _, frame = self.video.read()
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converts image frame to grayscale
        faces = face.detectMultiScale(grayFrame, 1.3, 5) # Detects face
        for (x, y, w, h) in faces:
            frameContent = grayFrame[y:y+h, x:x+w] # Make frame grayscale to match training images
            resized = cv2.resize(frameContent, (48, 48)) # Resize frame of face to match training images that model is used to
            prediction = model.predictEmotion(resized[np.newaxis, :, :, np.newaxis]) # Line that actually gets predicted emotion
            cv2.putText(frame, prediction, (x, y), font, 1, (255, 255, 0), 2) # Overlays prediction on video view
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) # Overlays rectangle around face
        return frame

def gen(camera): # If camera is being used to test that generate camera function
    while True:
        frame = camera.getFrame()
        cv2.imshow('Emotion Recognition',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

gen(VideoCamera())