import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

#load perceptron model
def load_perceptron_model():
    with open('perceptron_model.pkl', 'rb') as file:
        perceptron_model = pickle.load(file)
    return perceptron_model

#load backpropagation model
def load_backpropagationn_model():
    with open('backpropagation_spam_model.pkl', 'rb') as file:
        backpropagation_model = pickle.load(file)
    return backpropagation_model

# Load DNN model
def load_dnn_model():
    with open('dnn_model.pkl', 'rb') as file:
        dnn_model = pickle.load(file)
    return dnn_model

# Load RNN model
def load_rnn_model():
    with open('rnn_model.pkl', 'rb') as file:
        rnn_model = pickle.load(file)
    return rnn_model

# Load LSTM model
def load_lstm_model():
    with open('lstm_model.pkl', 'rb') as file:
        lstm_model = pickle.load(file)
    return lstm_model

# Load CNN model
def load_cnn_model():
    with open('cnn_model.pkl', 'rb') as file:
        cnn_model = pickle.load(file)
    return cnn_model

# Load perceptron tokeniser
def load_perceptron_tokeniser():
    with open('perceptron_tokeniser.pkl', 'rb') as file:
        tokeniser = pickle.load(file)
        return tokeniser
    
# Load backpropagation tokeniser
def load_backpropagation_tokeniser():
    with open('backpropagation_tokeniser.pkl', 'rb') as file:
        tokeniser = pickle.load(file)
    return tokeniser

# Load DNN tokeniser
def load_dnn_tokeniser():
    with open('dnn_tokeniser.pkl', 'rb') as file:
        tokeniser = pickle.load(file)
    return tokeniser

# Load RNN tokeniser
def load_rnn_tokeniser():
    with open('rnn_tokeniser.pkl', 'rb') as file:
        tokeniser = pickle.load(file)
    return tokeniser

# Load LSTM tokeniser
def load_lstm_tokeniser():
    with open('lstm_tokeniser.pkl', 'rb') as file:
        tokeniser = pickle.load(file)
    return tokeniser

# Streamlit App
st.title("Spam Classification and Tumor Detection")

# Choose between tasks
task = st.radio("Select Task", ("Spam Classification", "Tumor Detection"))

if task == "Spam Classification":
    # Input box for new message
    new_message = st.text_area("Enter a New Message:", value="")
    if st.button("Submit") and not new_message.strip():
        st.warning("Please enter a message.")

    if new_message.strip():
        st.subheader("Choose Model for Spam Classification")
        model_option = st.selectbox("Select Model", ("DNN", "RNN", "LSTM","Perceptron","BackPropagation"))

        # Load models and tokeniser dynamically based on the selected option
        if model_option == "Perceptron":
            model = load_perceptron_model()
            tokeniser = load_perceptron_tokeniser()
        elif model_option == "BackPropagation":
            model = load_backpropagationn_model()
            tokeniser = load_backpropagation_tokeniser()
        elif model_option == "DNN":
            model = load_dnn_model()
            tokeniser = load_dnn_tokeniser()
        elif model_option == "RNN":
            model = load_rnn_model()
            tokeniser = load_rnn_tokeniser()
        elif model_option == "LSTM":
            model = load_lstm_model()
            tokeniser = load_lstm_tokeniser()

        if st.button("Classify Spam"):
            max_length = 10
            new_message_tokens = tokeniser.texts_to_sequences([new_message])
            new_message_tokens = pad_sequences(new_message_tokens, maxlen=max_length, padding='post')
            prediction = model.predict(new_message_tokens)
            result = "Spam" if prediction > 0.5 else "Ham"

            st.subheader("Spam Classification Result")
            st.write(f"**{result}**")

elif task == "Tumor Detection":
    st.subheader("Tumor Detection")
    uploaded_file = st.file_uploader("Choose a tumor image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the tumor detection model
        cnn_model = load_cnn_model()
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=False, width=200)
        st.write("")

        if st.button("Detect Tumor"):
            img = Image.open(uploaded_file)
            img = img.resize((128, 128))
            img = np.array(img)
            input_img = np.expand_dims(img, axis=0)
            res = cnn_model.predict(input_img)
            result = "Tumor Detected" if res else "No Tumor Detected"

            st.subheader("Tumor Detection Result")
            st.write(f"**{result}**")
