import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from sklearn.model_selection import train_test_split
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_KEYWORDS = 4
NUM_INPUT_KEYWORDS = 20
LATENT_DIM = 10
NUM_EPOCHS = 24
BATCH_SIZE = 32

def sampling(args):
    mean, log_var = args
    epsilon = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * epsilon

def build_cvae_model(input_shape, output_shape, latent_dim):
    input_data = Input(shape=input_shape)
    output_data = Input(shape=output_shape)

    encoder_input = Dense(256, activation='relu')(input_data)
    mean = Dense(latent_dim)(encoder_input)
    log_var = Dense(latent_dim)(encoder_input)
    latent = Lambda(sampling, output_shape=(latent_dim,))([mean, log_var])

    output_encoder_input = Dense(256, activation='relu')(output_data)
    output_mean = Dense(latent_dim)(output_encoder_input)
    output_log_var = Dense(latent_dim)(output_encoder_input)
    output_latent = Lambda(sampling, output_shape=(latent_dim,))([output_mean, output_log_var])

    combined_latent = Concatenate()([latent, output_latent])

    hidden_decoder = Dense(256, activation='relu')(combined_latent)
    combined_output = Dense(output_shape[0], activation='sigmoid')(hidden_decoder)  # Change activation to 'sigmoid'

    cvae = Model([input_data, output_data], combined_output)

    reconstruction_loss = mse(output_data, combined_output)
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)
    output_kl_loss = -0.5 * tf.reduce_mean(1 + output_log_var - tf.square(output_mean) - tf.exp(output_log_var), axis=-1)
    cvae_loss = tf.reduce_mean(reconstruction_loss + kl_loss + output_kl_loss)

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
        recommended_outfit = outfit_dict["recommend_output"]

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

def load_test_data(test_data_path):
    with open(test_data_path, 'r') as json_file:
        test_data = json.load(json_file)
    return test_data

def main(test_data_path):
    training_data_path = "backend/data/mock_training_data/cvae_data_mock.json"
    with open(training_data_path, 'r') as json_file:
        json_data = json.load(json_file)

    encoded_input_outfits, encoded_output_outfits, input_keyword_to_index, output_keyword_to_index, input_index_to_keyword, output_index_to_keyword = prepare_conditional_training_data(json_data)

    input_train_data_split, input_val_data, output_train_data_split, output_val_data = train_test_split(encoded_input_outfits, encoded_output_outfits, test_size=0.2, random_state=42)

    cvae = build_cvae_model(input_shape=(NUM_INPUT_KEYWORDS,), output_shape=(NUM_INPUT_KEYWORDS,), latent_dim=LATENT_DIM)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.05)
    cvae.compile(optimizer=optimizer)
    cvae.fit([input_train_data_split, output_train_data_split], epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=([input_val_data, output_val_data], None),verbose=0)

    input_encoder = Input(shape=(NUM_INPUT_KEYWORDS,))
    encoder_hidden = cvae.layers[2](input_encoder)
    input_mean = Dense(LATENT_DIM)(encoder_hidden)
    input_log_var = Dense(LATENT_DIM)(encoder_hidden)
    input_latent = Lambda(sampling, output_shape=(LATENT_DIM,))([input_mean, input_log_var])
    encoder_model = Model(inputs=input_encoder, outputs=input_latent)

    input_decoder = Input(shape=(LATENT_DIM,))
    decoder_hidden = Dense(256, activation='relu')(input_decoder)  # Adjust the hidden layer dimensions as needed
    decoder_output = Dense(NUM_INPUT_KEYWORDS, activation='sigmoid')(decoder_hidden)
    decoder_model = Model(inputs=input_decoder, outputs=decoder_output)

    test_data = load_test_data(test_data_path)

    for idx, outfit_dict in enumerate(test_data):
        input_outfit = outfit_dict["Input fashion outfits"]
        user_data = outfit_dict["user_data"]
        social_trends = outfit_dict["social_trends"]
        input_combined_keywords = input_outfit + user_data + social_trends

        encoded_input = np.zeros((1, NUM_INPUT_KEYWORDS))
        for keyword in input_combined_keywords:
            if keyword in input_keyword_to_index:
                keyword_idx = input_keyword_to_index[keyword]
                if keyword_idx < NUM_INPUT_KEYWORDS:
                    encoded_input[0][keyword_idx] = 1

        latent_representation = encoder_model.predict(encoded_input,verbose=0)
        decoded_outfit = decoder_model.predict(latent_representation,verbose=0)

        num_top_keywords = 4
        top_keyword_indices = np.argsort(decoded_outfit[0])[::-1][:num_top_keywords]

        recommended_keywords = [output_index_to_keyword[i] for i in top_keyword_indices]
        
        return recommended_keywords

if __name__ == "__main__":
    main()