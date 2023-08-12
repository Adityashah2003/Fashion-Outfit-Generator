import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
import json

# Sample input data
num_keywords = 10  # Number of fashion items/keywords
num_epochs = 50
batch_size = 32

# VAE Model
latent_dim = 2

# Encoder
input_data = Input(shape=(num_keywords,))
hidden = Dense(256, activation='relu')(input_data)
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

# Sampling from latent space
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_input = Input(shape=(latent_dim,))
decoded = Dense(256, activation='relu')(decoder_input)
output_data = Dense(num_keywords, activation='sigmoid')(decoded)

# Define the VAE model
encoder = Model(input_data, z_mean)
decoder = Model(decoder_input, output_data)
vae = Model(input_data, decoder(z))

# Custom loss function to emphasize keyword reconstruction
def custom_vae_loss(y_true, y_pred):
    reconstruction_loss = mse(y_true, y_pred)
    reconstruction_loss *= num_keywords
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(reconstruction_loss + kl_loss)

# Compile the model with the custom loss
vae.compile(optimizer='adam', loss=custom_vae_loss)

# Generate or load your training data
training_data = np.random.randint(0, 2, size=(1000, num_keywords))  # Adjust the size according to your dataset

# Training
vae.fit(training_data, training_data, epochs=num_epochs, batch_size=batch_size)

# Load JSON and extract data
with open('data.json', 'r') as json_file:
    json_data = json.load(json_file)

input_sets = [
    json_data[0]['Input fashion outfits'],  # Update the indices accordingly
    json_data[1]['Input fashion outfits']   # Update the indices accordingly
]

for idx, input_set in enumerate(input_sets):
    user_input_strings = input_set
    encoded_user_inputs = []

    for input_string in user_input_strings:
        input_keywords = input_string.lower().split()[:num_keywords]  # Select a subset of keywords
        input_vector = [1 if keyword in input_keywords else 0 for keyword in range(num_keywords)]
        encoded_user_inputs.append(input_vector)

    encoded_user = encoder.predict(encoded_user_inputs)

    # Generate outfit recommendations
    decoded_outfits = decoder.predict(encoded_user)

    best_outfit_indices = np.argsort(decoded_outfits[0])[::-1]  # Indices of best items
    best_outfit = [input_set[i] for i in best_outfit_indices]

    print(f"Recommended outfit keywords for set {idx+1}:")
    print(best_outfit)
