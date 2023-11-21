import pickle
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from sklearn.preprocessing import minmax_scale

def elm_predict(X, W, b, round_output=False):
  Hinit = X @ W.T
  H = 1 / (1 + np.exp(-Hinit))
  y = H @ b

  if round_output:
    y = [np.round(x).astype(int) for x in y]

  return y

# Function to load the ELM model
def load_elm_model():
  with open('elm_model_hidden_neurons_5000.pkl', 'rb') as file:
    W, b = pickle.load(file)
  return W, b

# Function to preprocess the input image
def preprocess_image(image):
  # Resizing the image to the required dimensions
  image = image.resize((28, 28))
  # Convert to grayscale
  image = image.convert('L')
  # Convert to numpy array
  image_array = np.array(image)
  # Normalize pixel values
  image_array = minmax_scale(image_array)
  return image_array

def predict_digit(image, model):
  # Preprocess the input image
  processed_image = preprocess_image(image)
  # Add any additional preprocessing steps here

  # Flatten the image array
  flattened_image = processed_image.flatten()

  # Make predictions using the ELM model
  prediction = elm_predict(np.array([flattened_image]), model[0], model[1], round_output=True)

  return prediction[0]


# Load the ELM model
elm_model = load_elm_model()

# Streamlit app
st.title("Handwritten Digit Recognition")

# Create a drawable canvas for users to draw
canvas_result = st_canvas(
  fill_color="white",
  stroke_width=16,
  stroke_color="white",
  background_color="#000000",
  height=150,
  width=150,
  drawing_mode="freedraw",
  key="canvas",
)

if canvas_result.image_data is not None:
  # Convert the drawn image to a PIL Image
  drawn_image = Image.fromarray(canvas_result.image_data.astype(np.uint8))

  # Display the drawn image
  st.image(drawn_image, caption="Drawn Image", use_column_width=True)

  # Make prediction on the drawn image
  prediction = predict_digit(drawn_image, elm_model)
  st.success(f"Predicted Digit: {np.argmax(prediction)}")
