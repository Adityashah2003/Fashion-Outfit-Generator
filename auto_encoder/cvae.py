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
LATENT_DIM = 10
NUM_EPOCHS = 40
BATCH_SIZE = 32

def build_cvae_model(input_shape, output_shape, latent_dim):
    input_data = Input(shape=input_shape)
    output_data = Input(shape=output_shape)

    encoder_input = Dense(256, activation='relu')(input_data)
    mean = Dense(latent_dim)(encoder_input)
    log_var = Dense(latent_dim)(encoder_input)
    latent = Lambda(sampling)([mean, log_var])

    output_encoder_input = Dense(256, activation='relu')(output_data)
    output_mean = Dense(latent_dim)(output_encoder_input)
    output_log_var = Dense(latent_dim)(output_encoder_input)
    output_latent = Lambda(sampling)([output_mean, output_log_var])

    combined_latent = tf.keras.layers.concatenate([latent, output_latent])

    hidden_decoder = Dense(256, activation='relu')(combined_latent)
    combined_output = Dense(output_shape[0], activation='sigmoid')(hidden_decoder)

    cvae = Model([input_data, output_data], combined_output)

    reconstruction_loss = mse(output_data, combined_output)
    kl_loss = -0.5 * tf.keras.backend.mean(1 + log_var - tf.keras.backend.square(mean) - tf.keras.backend.exp(log_var), axis=-1)
    output_kl_loss = -0.5 * tf.keras.backend.mean(1 + output_log_var - tf.keras.backend.square(output_mean) - tf.keras.backend.exp(output_log_var), axis=-1)
    cvae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss + output_kl_loss)

    cvae.add_loss(cvae_loss)
    return cvae

def prepare_conditional_training_data(lists_data):
    input_keyword_set = set()
    output_keyword_set = set()
    input_outfits = []
    output_outfits = []

    for outfit_dict in lists_data:
        input_outfit = outfit_dict["Input fashion outfits"]
        user_data = outfit_dict["user_data"]
        social_trends = outfit_dict["social_trends"]
        recommended_outfit = outfit_dict["recommended_outfit_keywords"]

        input_combined_keywords = input_outfit + user_data + social_trends
        input_outfits.append(input_combined_keywords)
        input_keyword_set.update(input_combined_keywords)

        output_outfits.append(recommended_outfit)
        output_keyword_set.update(recommended_outfit)

    input_keyword_to_index = {keyword: idx for idx, keyword in enumerate(input_keyword_set)}
    input_index_to_keyword = {idx: keyword for keyword, idx in input_keyword_to_index.items()}

    output_keyword_to_index = {keyword: idx for idx, keyword in enumerate(output_keyword_set)}
    output_index_to_keyword = {idx: keyword for keyword, idx in output_keyword_to_index.items()}

    encoded_input_outfits = np.zeros((len(input_outfits), NUM_INPUT_KEYWORDS))
    encoded_output_outfits = np.zeros((len(output_outfits), NUM_INPUT_KEYWORDS))

    for i, (input_outfit, output_outfit) in enumerate(zip(input_outfits, output_outfits)):
        for keyword in input_outfit:
            if keyword in input_keyword_to_index:
                keyword_idx = input_keyword_to_index[keyword]
                if keyword_idx < NUM_INPUT_KEYWORDS:
                    encoded_input_outfits[i][keyword_idx] = 1

        for keyword in output_outfit:
            if keyword in output_keyword_to_index:
                keyword_idx = output_keyword_to_index[keyword]
                if keyword_idx < NUM_INPUT_KEYWORDS:
                    encoded_output_outfits[i][keyword_idx] = 1

    return encoded_input_outfits, encoded_output_outfits, input_keyword_to_index, output_keyword_to_index, input_index_to_keyword, output_index_to_keyword

# ... (other parts of the main function)

def main():
    file_path = "C:\\Users\\Aditya\\Documents\\GitHub\\FOG\\data\\mock_final_data.json"
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)

    input_train_data, output_train_data, input_keyword_to_index, output_keyword_to_index, input_index_to_keyword, output_index_to_keyword = prepare_conditional_training_data(json_data)

    cvae = build_cvae_model(input_shape=(NUM_INPUT_KEYWORDS,), output_shape=(NUM_INPUT_KEYWORDS,), latent_dim=LATENT_DIM)
    train_data, val_data = train_test_split((input_train_data, output_train_data), test_size=0.2, random_state=42)

    cvae.compile(optimizer='adam')
    cvae.fit(train_data, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_data, None))

    encoder = Model(cvae.input[0], cvae.get_layer('lambda').output)
    output_encoder = Model(cvae.input[1], cvae.get_layer('lambda_1').output)
    combined_encoder = Model(cvae.input[0], cvae.get_layer('lambda_2').output)

    decoder = Model(cvae.input[0], cvae.output)

    for idx, _ in enumerate(json_data):
        input_vector = input_train_data[idx].reshape(1, -1)
        output_vector = output_train_data[idx].reshape(1, -1)

        combined_latent_representation = combined_encoder.predict([input_vector, output_vector])  # Use both input and output vectors

        decoded_outfit = decoder.predict(input_vector)

        best_outfit_indices = np.argsort(decoded_outfit[0])[::-1][:NUM_KEYWORDS]
        best_outfit = [output_index_to_keyword[idx] for idx in best_outfit_indices]

        print(f"Recommended outfit keywords for list {idx + 1}:")
        print(best_outfit)

if __name__ == "__main__":
    main()

