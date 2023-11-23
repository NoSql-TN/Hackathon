import numpy as np 
import pandas as pd
import tensorflow as tf
import sys
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

'''from python.database.get_db import get_db3

db3 = get_db3()
file3 = pd.DataFrame(db3)
file3 = file3.dropna()
file3 = file3.drop_duplicates()

file3 = file3[(file3['track_genre'].isin(['hip-hop', 'emo', 'rock', 'dance', 'pop'])) & (file3['parole'] != 'None')]

common_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'during', 'to', 'What', 'Which', 'Is', 'If', 'While', 'This']
    
def count_common_words(text):
    common_count = 0
    words = text.split()
    for word in words:
        if word in common_words:
            common_count += 1
    return common_count

file3['common_word_count'] = file3['parole'].apply(lambda x: count_common_words(x))
file3_filtered = file3[file3['common_word_count'] >= 20]

file3_filtered = file3_filtered[file3_filtered['parole'].str.contains('nasha chahida sharab thoda mangala janaab') == False]
file3_filtered = file3_filtered[file3_filtered['parole'].str.contains('Bom diggy diggy bom bom') == False]
file3_filtered = file3_filtered[file3_filtered['parole'].str.contains('Sidhu Moose Wala') == False]
file3_filtered = file3_filtered[file3_filtered['parole'].str.contains('I will bring the fire And my name keeps going worldwide So my job is to satisfy') == False]
    
        
lyrics = file3_filtered['parole'].tolist()
lyrics_text = [lyric.strip() for lyric in lyrics if lyric.strip() != ""]

def train_gpt2_on_lyrics():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = TFGPT2LMHeadModel.from_pretrained('gpt2', from_pt=True)

    # Tokenize the dataset
    input_ids = tokenizer.encode(lyrics_text, return_tensors='tf')

    # Define model training configuration
    train_config = tf.data.Dataset.from_tensor_slices(input_ids)
    train_config = train_config.shuffle(buffer_size=1024).batch(4, drop_remainder=True)

    # Fine-tuning the model on your lyrics dataset
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer])

    # Start training
    model.fit(train_config, epochs=3)  # You can adjust the number of epochs

    # Save the trained model
    output_dir = 'gpt-model'  # Replace with desired output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

'''

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("caseyhahn/gpt2-finetuned-genius-lyrics-updated-data")
model = AutoModelForCausalLM.from_pretrained("caseyhahn/gpt2-finetuned-genius-lyrics-updated-data", from_tf=True)

def generate_lyrics(seed_text, num_return_sequences=1, max_length=100):
    five_lyrics = []
    for i in range(5):
        input_ids = tokenizer.encode(seed_text, return_tensors='pt')
        output = model.generate(input_ids=input_ids, max_length=max_length, num_return_sequences=num_return_sequences, temperature=1.0, repetition_penalty=1.0, do_sample=True)
        five_lyrics.append(tokenizer.decode(output[0], skip_special_tokens=True))
    with open('lyrics.txt', 'w') as f:
        for item in five_lyrics:
            f.write("%s\n" % item)
            f.write("--sep--\n")
    print("Done!")

if __name__ == "__main__":
    lyrics = ""
    for arg in sys.argv[1:]:
        lyrics += arg + " "
    generate_lyrics(lyrics)