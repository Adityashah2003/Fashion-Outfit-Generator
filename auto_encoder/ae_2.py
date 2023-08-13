import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from sklearn.model_selection import train_test_split
import json

# Constants
NUM_KEYWORDS = 4
NUM_INPUT_KEYWORDS = 20
LATENT_DIM = 10  # Dimension of latent space
NUM_EPOCHS = 40
BATCH_SIZE = 32

# Model Architecture
def build_vae_model(input_shape, latent_dim):
    input_data = Input(shape=input_shape)
    hidden_encoder = Dense(256, activation='relu')(input_data)
    mean = Dense(latent_dim)(hidden_encoder)
    log_var = Dense(latent_dim)(hidden_encoder)

    def sampling(args):
        mean, log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(mean)[0], latent_dim))
        return mean + tf.keras.backend.exp(0.5 * log_var) * epsilon

    latent = Lambda(sampling)([mean, log_var])

    hidden_decoder = Dense(256, activation='relu')(latent)
    output_data = Dense(input_shape[0], activation='sigmoid')(hidden_decoder)

    vae = Model(input_data, output_data)

    reconstruction_loss = mse(input_data, output_data)
    kl_loss = -0.5 * tf.keras.backend.mean(1 + log_var - tf.keras.backend.square(mean) - tf.keras.backend.exp(log_var), axis=-1)
    vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
    
    vae.add_loss(vae_loss)
    return vae

def prepare_training_data(lists_data):
    keyword_set = set()
    outfits = []

    for outfit_dict in lists_data:
        input_outfit = outfit_dict["Input fashion outfits"]
        user_data = outfit_dict["user_data"]
        social_trends = outfit_dict["social_trends"]
        
        combined_keywords = input_outfit + user_data + social_trends
        outfits.append(combined_keywords)
        keyword_set.update(combined_keywords)

    keyword_to_index = {keyword: idx for idx, keyword in enumerate(keyword_set)}
    index_to_keyword = {idx: keyword for keyword, idx in keyword_to_index.items()}

    encoded_outfits = np.zeros((len(outfits), NUM_INPUT_KEYWORDS))

    for i, outfit in enumerate(outfits):
        for keyword in outfit:
            if keyword in keyword_to_index:
                keyword_idx = keyword_to_index[keyword]
                if keyword_idx < NUM_INPUT_KEYWORDS:
                    encoded_outfits[i][keyword_idx] = 1

    return encoded_outfits, keyword_to_index, index_to_keyword


def main():
    file_path = "C:\\Users\\Aditya\\Documents\\GitHub\\FOG\\data\\mock_final_data.json"
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)

    combined_keywords_list = []
    for entry in json_data:
        input_keywords = entry['Input fashion outfits']
        user_keywords = entry['user_data']
        social_keywords = entry['social_trends']

        combined_keywords = input_keywords + user_keywords + social_keywords
        combined_keywords_list.append(combined_keywords)  # Store combined keywords for this outfit separately

        # print(f"Combined keywords for current outfit: {combined_keywords}")

    training_data, keyword_to_index, index_to_keyword = prepare_training_data(json_data)

    vae = build_vae_model(input_shape=(NUM_INPUT_KEYWORDS,), latent_dim=LATENT_DIM)
    train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)

    vae.compile(optimizer='adam')
    vae.fit(train_data, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_data, None))

    encoder = Model(vae.input, vae.get_layer('lambda').output)
    decoder = Model(vae.input, vae.output)

    for idx, _ in enumerate(json_data):
        input_vector = training_data[idx].reshape(1, -1)
        latent_representation = encoder.predict(input_vector)

        decoded_outfit = decoder.predict(input_vector)

        best_outfit_indices = np.argsort(decoded_outfit[0])[::-1][:NUM_KEYWORDS]
        best_outfit = [index_to_keyword[idx] for idx in best_outfit_indices]
        
        print(f"Recommended outfit keywords for list {idx + 1}:")
        print(best_outfit)

if __name__ == "__main__":
    main()

