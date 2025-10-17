import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # last n tokens
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)
    return tokenizer.index_word.get(predicted_word_index, "<unknown>")

def predict_multiple_words(model, tokenizer, text, max_sequence_len, num_words=15):
    for _ in range(num_words):
        next_word = predict_next_word(model, tokenizer, text, max_sequence_len)
        text += "," + next_word
    return text

def predict_next_word_temp(model, tokenizer, text, max_sequence_len, temperature=1.0):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)[0]

    # Use temperature sampling for diversity
    preds = np.log(predicted + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    predicted_word_index = np.random.choice(len(preds), p=preds)
    
    return tokenizer.index_word.get(predicted_word_index, "<unknown>")

# Streamlit app
st.title("Next Word Prediction with LSTM")
st.write("Enter a sequence of words, and the model will predict the most likely next word.")

input_text = st.text_input("Enter a sequence of words:", "To be or not to")

if st.button("Predict Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter some words to predict.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word_temp(model, tokenizer, input_text, max_sequence_len,0.8)
        st.success(f"**Next word:** {next_word}")
