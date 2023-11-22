import numpy as np 
import pandas as pd
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from python.database.get_db import get_db2

db2 = get_db2()
file2 = pd.DataFrame(db2)
file2 = file2.dropna()
file2 = file2.drop_duplicates()

def generator():
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Add a padding token to the tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # You can change '[PAD]' to any token you prefer

    model = TFGPT2LMHeadModel.from_pretrained(model_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)

    chunk_size = 100  # Define the chunk size for training

    for i in range(0, len(file2), chunk_size):
        chunk = file2.iloc[i:i+chunk_size]['lyrics'].values.tolist()
        text = "\n".join(chunk)

        # Tokenize the chunk of text
        tokenized_text = tokenizer(text, return_tensors='tf', padding=True, truncation=True)

        # Train the model on the current chunk
        model.train_on_batch(tokenized_text, tokenized_text)  # Using train_on_batch for fine-tuning

    # Generate lyrics based on a prompt
    prompt = "I'm feeling happy today because"
    input_ids = tokenizer.encode(prompt, return_tensors='tf')
    generated_lyrics = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1)
    decoded_lyrics = tokenizer.decode(generated_lyrics[0], skip_special_tokens=True)
    print(decoded_lyrics)