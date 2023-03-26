import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

st.title("Animal Classification in Video Input")

# Load the pre-trained classification model
model = load_model("animal_classifier.h5")

# Define the labels for the classification classes
labels = ["cat", "dog", "elephant", "giraffe", "horse", "sheep", "cow", "squirrel"]

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))

# Define a function to preprocess and classify the image
def classify_image(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Predict the class probabilities for the input image
    preds = model.predict(image)
    idx = np.argmax(preds)
    label = labels[idx]
    return label

# Run the app
while True:
    # Read the current frame from the video capture object
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if ret:
        # Classify the current frame
        label = classify_image(frame)

        # Draw the classification label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the frame to the output video file
        out.write(frame)

        # Display the frame in the Streamlit app
        st.image(frame, channels="BGR")

        # Check if the user has clicked the "Stop" button
        if st.button("Stop"):
            break
    else:
        break

# Release the video capture object and the output video writer object
cap.release()
out.release()

# Deactivate the virtual environment
deactivate
