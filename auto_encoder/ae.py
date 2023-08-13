import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import json

# Constants
NUM_KEYWORDS = 10
NUM_INPUT_KEYWORDS = 20
NUM_EPOCHS = 50
BATCH_SIZE = 32

# Model Architecture
def build_vae_model(input_shape):
    input_data = Input(shape=input_shape)
    hidden = Dense(512, activation='relu')(input_data)
    output_data = Dense(NUM_KEYWORDS, activation='sigmoid')(hidden)  # Use sigmoid for binary classification

    vae = Model(input_data, output_data)
    vae.compile(optimizer='adam', loss='binary_crossentropy')  # Use binary cross-entropy for multi-label classification
    return vae

# Data Preprocessing
def prepare_training_data(json_data):
    keyword_set = set()

    outfits = []
    for entry in json_data:
        input_set = entry['Input fashion outfits']
        user_data = entry['user_data']
        social_trends = entry['social_trends']

        input_keywords = input_set + user_data + social_trends
        outfits.append(input_keywords)
        keyword_set.update(input_keywords)

    keyword_to_index = {keyword: idx for idx, keyword in enumerate(keyword_set)}
    index_to_keyword = {idx: keyword for keyword, idx in keyword_to_index.items()}  # Add this line

    encoded_outfits = np.zeros((len(outfits), NUM_KEYWORDS))

    for i, outfit in enumerate(outfits):
        for keyword in outfit:
            keyword_idx = keyword_to_index[keyword]
            encoded_outfits[i][keyword_idx] = 1

    return encoded_outfits, keyword_to_index, index_to_keyword

# Main function
def main():
    vae = build_vae_model(input_shape=(NUM_INPUT_KEYWORDS + len(["user_data", "social_trends"]),))
    with open('fashion_data.json', 'r') as json_file:
        json_data = json.load(json_file)
        
    training_data, keyword_to_index, index_to_keyword = prepare_training_data(json_data)

    train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)

    vae.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # Encoder and Decoder Models
    encoder = Model(vae.input, vae.layers[-2].output)  # Define the encoder model
    decoder_input = Input(shape=(NUM_KEYWORDS,))
    decoder_layer = vae.layers[-1](decoder_input)
    decoder = Model(decoder_input, decoder_layer)

    # Recommendation logic
    for idx, entry in enumerate(json_data):
        input_set = entry['Input fashion outfits']
        user_data = entry['user_data']
        social_trends = entry['social_trends']

        input_keywords = input_set + user_data + social_trends
        input_vector = np.zeros(NUM_INPUT_KEYWORDS + len(["user_data", "social_trends"]))
        
        for keyword in input_keywords:
            keyword_idx = keyword_to_index[keyword]  # Map keyword to index
            input_vector[keyword_idx] = 1
        
        encoded_user = encoder.predict(np.array([input_vector]))
        decoded_outfits = decoder.predict(encoded_user)

        best_outfit_indices = np.argsort(decoded_outfits[0])[::-1][:NUM_KEYWORDS]  # Indices of best items
        best_outfit = [index_to_keyword[idx] for idx in best_outfit_indices]

        print(f"Recommended outfit keywords for set {idx + 1}:")
        print(best_outfit)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
