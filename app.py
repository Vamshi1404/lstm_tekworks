import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="LSTM Text Generator",
    page_icon="📝",
    layout="centered"
)

model = tf.keras.models.load_model(
    "models/lstm_text_generator.keras"
)

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/max_sequence_len.pkl", "rb") as f:
    max_sequence_len = pickle.load(f)

index_to_word = {
    v: k for k, v in tokenizer.word_index.items()
}

st.title("📝 Shakespeare Text Generator")

seed_text = st.text_input(
    "Enter Starting Text",
    "once upon a"
)

num_words = st.slider(
    "Number of Words",
    5,
    100,
    20
)

def generate_text(seed_text, num_words):

    generated = seed_text

    for _ in range(num_words):

        token_list = tokenizer.texts_to_sequences(
            [generated]
        )[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        prediction = model.predict(
            token_list,
            verbose=0
        )

        predicted_index = np.argmax(
            prediction,
            axis=-1
        )[0]

        next_word = index_to_word.get(
            predicted_index,
            ""
        )

        generated += " " + next_word

    return generated

if st.button("Generate"):

    result = generate_text(
        seed_text,
        num_words
    )

    st.subheader("Generated Text")
    st.write(result)