"""
an adversarial autoencoder for raw audio

Notes on discriminators:
the model uses two discriminators, one for distinguishing between real and fake prior distributions
and one for distinguishing between real and fake audio.

Notes on dilated convolutions:
the model combines dilated convolution with regular convolution.
It does so my creating branched paths for each convolution step where dilation is greater than 0,
the branched paths then joint again by adding the results of the regular convolution and dilated convolution.
"""

# Imports

import tensorflow as tf
import tensorflow.keras as keras
import simpleaudio as sa
from scipy.io.wavfile import read as audio_read
from scipy.io.wavfile import write as audio_write
import numpy as np
import glob
from matplotlib import pyplot as plt
import os, time
import json
from common import utils

# Configurations

# Audio Configuration

audio_file_path = "../data/sonification_audio"
#audio_file_path = "../data/improvisation_audio"
audio_save_sample_rate = 32000 # sample rate for saving example sonifications
audio_sample_rate = 32000 # numer of audio samples per sec
audio_window_length = 256 # size of audio window in number of samples
audio_window_rate = 250 # number of audio windows per sec
standardize_audio = True

# Training Configuration
audio_window_offset = 16
batch_size = 128
epochs = 400
train_percentage = 0.9
test_percentage = 0.1
ae_learning_rate = 1e-4
disc_audio_learning_rate = 4e-4
disc_prior_learning_rate = 4e-4
ae_rec_loss_scale = 1.0
ae_disc_audio_loss_scale = 1.0
ae_disc_prior_loss_scale = 1.0
ae_l1_loss_scale = 0.0
ae_l2_loss_scale = 0.01
disc_prior_label_smoothing = 0.9
disc_audio_label_smoothing = 0.9
model_save_interval = 50
save_history = False

# Model Configuration

# Autoencoder Configuration
latent_dim = 32
ae_activation_function = "relu"
ae_conv_filter_counts = [32, 64, 128, 256]
ae_conv_kernel_sizes = [7, 7, 7, 7]
ae_conv_strides = [4, 4, 4, 4]
ae_conv_dilations = [30, 9, 2, 0]
ae_dense_layer_sizes = [32]
ae_dropout_prob = 0.0
ae_use_batch_normalization = True
ae_use_layer_normalization = False

# Prior Discriminator Configuration
disc_prior_activation_function = "leaky_relu"
disc_prior_dense_layer_sizes = [32, 32]
disc_prior_dropout_prob = 0.0
disc_prior_use_batch_normalization = True
disc_prior_use_layer_normalization = False

# Audio Discriminator Configuration
disc_audio_activation_function = "leaky_relu"
disc_audio_conv_filter_counts = [32, 64, 128, 256]
disc_audio_conv_kernel_sizes = [7, 7, 7, 7]
disc_audio_conv_strides = [4, 4, 4, 4]
disc_audio_conv_dilations = [30, 9, 2, 0]
disc_audio_dense_layer_sizes = [32]
disc_audio_dropout_prob = 0.0
disc_audio_use_batch_normalization = True
disc_audio_use_layer_normalization = False

# Save / Load Model Weights
save_models = False
save_weights = False
load_weights = True
disc_prior_weights_file = "aae/weights/sonification/disc_prior_weights_epoch_300"
disc_audio_weights_file = "aae/weights/sonification/disc_audio_weights_epoch_300"
ae_encoder_weights_file = "aae/weights/sonification/ae_encoder_weights_epoch_300"
ae_decoder_weights_file = "aae/weights/sonification/ae_decoder_weights_epoch_300"
ae_weights_file = "aae/weights/sonification/ae_weights_epoch_300"

# Save Audio Examples
save_audio = False
audio_save_interval = 100
audio_save_start_times = [ 40.0, 120.0, 300.0, 480.0  ] # in seconds
audio_save_duration = 10.0
audio_traverse_start_window = 0
audio_traverse_end_window = 1000
audio_traverse_window_count = 10
audio_traverse_interpolation_count = 100


# Prepare Dataset

# load audio 
audio = audio_read(audio_file_path + ".wav")

# convert audio from int16 to normalized float
audio_sample_scale = 2**15
audio_float_array = np.array(audio[1], dtype=np.float32)
audio_float_array = audio_float_array / audio_sample_scale

# create standardized audio float array
if standardize_audio:
    audio_mean = np.mean(audio_float_array)
    audio_std = np.std(audio_float_array)
    audio_standardized = (audio_float_array - audio_mean) / (audio_std)
else:
    audio_standardized = audio_float_array

# gather audio training data
audio_training_data = []
max_audio_idx = audio_standardized.shape[0] - audio_window_length - 1
for audio_idx in range(0, max_audio_idx, audio_window_offset):
    audio_training_data.append( audio_standardized[audio_idx:audio_idx+audio_window_length] )
audio_training_data = np.array(audio_training_data)
audio_count = audio_training_data.shape[0]

# create dataset
dataset = tf.data.Dataset.from_tensor_slices(audio_training_data)
dataset = dataset.shuffle(audio_count).batch(batch_size, drop_remainder=True)
dataset_size = audio_count // batch_size

# train / test split
train_size = int(train_percentage * dataset_size)
test_size = int(test_percentage * dataset_size)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Save Original Audio Examples
def create_ref_sonifications(start_time, duration, file_name):
    start_sample = int(start_time * audio_save_sample_rate)
    sample_count = int(duration * audio_save_sample_rate)
    ref_audio = audio_standardized[start_sample:start_sample + sample_count]
    if standardize_audio:
        ref_audio = audio_mean + ref_audio * audio_std
    audio_write(file_name, audio_save_sample_rate, ref_audio)

if save_audio:
    for audio_start_time in audio_save_start_times:
        file_name = "../results/aae/ref_audio_{}.wav".format(audio_start_time)
        create_ref_sonifications(audio_start_time, audio_save_duration, file_name)

# Create Models

# Create Prior Discriminator

def sample_normal(shape):
    return np.random.normal(size=shape)

disc_prior_input = keras.layers.Input(shape=(latent_dim))
x = disc_prior_input
for layer_size in disc_prior_dense_layer_sizes:    
    if disc_prior_use_batch_normalization:
        x = keras.layers.BatchNormalization()(x)
    if disc_prior_activation_function == "leaky_relu":
        x = keras.layers.Dense(layer_size)(x)
        x = keras.layers.LeakyReLU(alpha = 0.01)(x)
    else:
        x = keras.layers.Dense(layer_size, activation=disc_prior_activation_function)(x)
    if disc_prior_use_layer_normalization:
        x = keras.layers.LayerNormalization()(x)
    if disc_prior_dropout_prob > 0.0:
       x = tf.keras.layers.Dropout(disc_prior_dropout_prob)(x)
disc_prior_output = keras.layers.Dense(1, activation="sigmoid")(x)
    
disc_prior = keras.Model(disc_prior_input, disc_prior_output)
disc_prior.summary()

if save_models == True:
    disc_prior.save("aae/models/disc_prior")
    keras.utils.plot_model(disc_prior, show_shapes=True, dpi=64, to_file='aae/models/disc_prior.png')

if load_weights and disc_prior_weights_file:
    disc_prior.load_weights(disc_prior_weights_file)

# Create Audio Discriminator

disc_audio_in = keras.layers.Input(shape=(audio_window_length))
x = disc_audio_in

if len(disc_audio_conv_filter_counts) > 0:
    x = tf.expand_dims(x, axis=2)
    for filter_count, kernel_size, stride, dilation in zip(disc_audio_conv_filter_counts, disc_audio_conv_kernel_sizes, disc_audio_conv_strides, disc_audio_conv_dilations):
        if disc_audio_use_batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if dilation > 0:
            if disc_audio_activation_function == "leaky_relu":
                x1 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, padding="same")(x)
                x1 = keras.layers.LeakyReLU(alpha = 0.01)(x1)
                x2 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, dilation_rate=dilation, padding="same")(x)
                x2 = keras.layers.LeakyReLU(alpha = 0.01)(x2)
                x2 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, padding="same")(x2)
                x2 = keras.layers.LeakyReLU(alpha = 0.01)(x2)
                x = keras.layers.Add()([x1, x2])
            else:
                x1 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, activation=disc_audio_activation_function, padding="same")(x)
                x2 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, dilation_rate=dilation, activation=disc_audio_activation_function, padding="same")(x)
                x2 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, activation=disc_audio_activation_function, padding="same")(x2)
                x = keras.layers.Add()([x1, x2])
        else:
            if disc_audio_activation_function == "leaky_relu":
                x = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, padding="same")(x)
                x = keras.layers.LeakyReLU(alpha = 0.01)(x)
            else:
                x = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, activation=disc_audio_activation_function, padding="same")(x)
        if disc_audio_use_layer_normalization:
            x = keras.layers.LayerNormalization()(x)
        if disc_audio_dropout_prob > 0.0:
           x = tf.keras.layers.Dropout(disc_audio_dropout_prob)(x)

x = tf.keras.layers.Flatten()(x)
if len(disc_audio_dense_layer_sizes) > 0:
    for layer_size in disc_audio_dense_layer_sizes:
        if disc_audio_use_batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if disc_audio_activation_function == "leaky_relu":
            x = keras.layers.Dense(layer_size)(x)
            x = keras.layers.LeakyReLU(alpha = 0.01)(x)
        else:
            x = keras.layers.Dense(layer_size, activation=disc_audio_activation_function)(x)
        if disc_audio_use_layer_normalization:
            x = keras.layers.LayerNormalization()(x)
        if disc_audio_dropout_prob > 0.0:
           x = tf.keras.layers.Dropout(disc_audio_dropout_prob)(x)

if disc_audio_use_batch_normalization:
    x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(units = 1)(x)

disc_audio_out = x

disc_audio = keras.Model(disc_audio_in, disc_audio_out)
disc_audio.summary()

if save_models == True:
    disc_audio.save("aae/models/disc_audio")
    keras.utils.plot_model(disc_audio, show_shapes=True, dpi=64, to_file='aae/models/disc_audio.png')

if load_weights and disc_audio_weights_file:
    disc_audio.load_weights(disc_audio_weights_file)

# Create Audio Encoder

ae_encoder_in = keras.layers.Input(shape=(audio_window_length))
x = ae_encoder_in

# encoder convolution layers
if len(ae_conv_filter_counts) > 0:
    x = tf.expand_dims(x, axis=2)
    for filter_count, kernel_size, stride, dilation in zip(ae_conv_filter_counts, ae_conv_kernel_sizes, ae_conv_strides, ae_conv_dilations):
        if ae_use_batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if dilation > 0:
            if ae_activation_function == "leaky_relu":
                x1 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, padding="same")(x)
                x1 = keras.layers.LeakyReLU(alpha = 0.01)(x1)
                x2 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, dilation_rate=dilation, padding="same")(x)
                x2 = keras.layers.LeakyReLU(alpha = 0.01)(x2)
                x2 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, padding="same")(x2)
                x2 = keras.layers.LeakyReLU(alpha = 0.01)(x2)
                x = keras.layers.Add()([x1, x2])
            else:
                x1 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, activation=ae_activation_function, padding="same")(x)
                x2 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, dilation_rate=dilation, activation=ae_activation_function, padding="same")(x)
                x2 = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, activation=ae_activation_function, padding="same")(x2)
                x = keras.layers.Add()([x1, x2])
        else:
            if disc_audio_activation_function == "leaky_relu":
                x = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, padding="same")(x)
                x = keras.layers.LeakyReLU(alpha = 0.01)(x)
            else:
                x = keras.layers.Conv1D(filters=filter_count, kernel_size=kernel_size, strides=stride, activation=disc_audio_activation_function, padding="same")(x)
        if ae_use_layer_normalization:
            x = keras.layers.LayerNormalization()(x)
        if ae_dropout_prob > 0.0:
           x = tf.keras.layers.Dropout(ae_dropout_prob)(x)
    
    shape_before_flattening = x.shape[1:]
    x = keras.layers.Flatten()(x)

shape_after_flattening = x.shape[1:]

# encoder dense layers
if len(ae_dense_layer_sizes) > 0:
    for layer_size in ae_dense_layer_sizes:
        if ae_use_batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if ae_activation_function == "leaky_relu":
            x = keras.layers.Dense(layer_size)(x)
            x = keras.layers.LeakyReLU(alpha = 0.01)(x)
        else:
            x = keras.layers.Dense(layer_size, activation=ae_activation_function)(x)
        if ae_use_layer_normalization:
            x = keras.layers.LayerNormalization()(x)
        if ae_dropout_prob > 0.0:
           x = tf.keras.layers.Dropout(ae_dropout_prob)(x)

x = keras.layers.Dense(latent_dim)(x)
ae_encoder_out = x

ae_encoder = keras.Model(inputs=ae_encoder_in, outputs=ae_encoder_out)
ae_encoder.summary()

if save_models == True:
    ae_encoder.save("aae/models/ae_encoder")
    keras.utils.plot_model(ae_encoder, show_shapes=True, dpi=64, to_file='aae/models/ae_encoder.png')

if load_weights and ae_encoder_weights_file:
    ae_encoder.load_weights(ae_encoder_weights_file)


# Create Audio Decoder

rev_ae_dense_layer_sizes = ae_dense_layer_sizes.copy()
rev_ae_dense_layer_sizes.reverse()

rev_ae_conv_filter_counts = ae_conv_filter_counts.copy()
rev_ae_conv_kernel_sizes = ae_conv_kernel_sizes.copy()
rev_ae_conv_strides = ae_conv_strides.copy()
rev_ae_conv_dilations = ae_conv_dilations.copy()

rev_ae_conv_filter_counts.reverse()
rev_ae_conv_kernel_sizes.reverse()
rev_ae_conv_strides.reverse()
rev_ae_conv_dilations.reverse()

rev_ae_conv_filter_counts = rev_ae_conv_filter_counts[1:]
rev_ae_conv_filter_counts.append(1)

ae_decoder_in = keras.layers.Input(shape=(latent_dim))
x = ae_decoder_in

if len(rev_ae_dense_layer_sizes) > 0:
    for layer_size in rev_ae_dense_layer_sizes:
        if ae_use_batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if ae_activation_function == "leaky_relu":
            x = keras.layers.Dense(layer_size)(x)
            x = keras.layers.LeakyReLU(alpha = 0.01)(x)
        else:
            x = keras.layers.Dense(layer_size, activation=ae_activation_function)(x)
        if ae_use_layer_normalization:
            x = keras.layers.LayerNormalization()(x)
        if ae_dropout_prob > 0.0:
           x = tf.keras.layers.Dropout(ae_dropout_prob)(x)
           
if ae_use_batch_normalization:
    x = keras.layers.BatchNormalization()(x)
if len(ae_conv_filter_counts) > 0:
    if ae_activation_function == "leaky_relu":
        x = keras.layers.Dense(units=shape_after_flattening[0])(x)
        x = keras.layers.LeakyReLU(alpha = 0.01)(x)
    else:
        x = keras.layers.Dense(units=shape_after_flattening[0], activation = ae_activation_function)(x)
else:
    x = keras.layers.Dense(units=shape_after_flattening[0], activation = "linear")(x)
if ae_use_layer_normalization:
    x = keras.layers.LayerNormalization()(x)

if len(ae_conv_filter_counts) > 0:
    x = keras.layers.Reshape(shape_before_flattening)(x)
    
    for filter_count, kernel_size, stride, dilation in zip( rev_ae_conv_filter_counts[:-1], rev_ae_conv_kernel_sizes[:-1], rev_ae_conv_strides[:-1], rev_ae_conv_dilations[:-1]):
        if ae_use_batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if dilation > 0:
            if ae_activation_function == "leaky_relu":
                x1 = keras.layers.Conv1DTranspose(filters=filter_count, kernel_size=kernel_size, strides=stride, padding="same")(x)
                x1 = keras.layers.LeakyReLU(alpha = 0.01)(x1)
                x2 = keras.layers.Conv1DTranspose(filters=filter_count, kernel_size=kernel_size, strides=stride, padding="same")(x)
                x2 = keras.layers.LeakyReLU(alpha = 0.01)(x2)
                x2 = keras.layers.Conv1DTranspose(filters=filter_count, kernel_size=kernel_size, dilation_rate=dilation, padding="same")(x2)
                x2 = keras.layers.LeakyReLU(alpha = 0.01)(x2)
                x = keras.layers.Add()([x1, x2])
            else:
                x1 = keras.layers.Conv1DTranspose(filters=filter_count, kernel_size=kernel_size, strides=stride, activation=ae_activation_function, padding="same")(x)
                x2 = keras.layers.Conv1DTranspose(filters=filter_count, kernel_size=kernel_size, strides=stride, activation=ae_activation_function, padding="same")(x)
                x2 = keras.layers.Conv1DTranspose(filters=filter_count, kernel_size=kernel_size, dilation_rate=dilation, activation=ae_activation_function, padding="same")(x2)
                x = keras.layers.Add()([x1, x2])
        else:
            if disc_audio_activation_function == "leaky_relu":
                x = keras.layers.Conv1DTranspose(filters=filter_count, kernel_size=kernel_size, strides=stride, padding="same")(x)
                x = keras.layers.LeakyReLU(alpha = 0.01)(x)
            else:
                x = keras.layers.Conv1DTranspose(filters=filter_count, kernel_size=kernel_size, strides=stride, activation=disc_audio_activation_function, padding="same")(x)
        if ae_use_layer_normalization:
            x = keras.layers.LayerNormalization()(x)
        if ae_dropout_prob > 0.0:
           x = tf.keras.layers.Dropout(ae_dropout_prob)(x)

    if ae_use_batch_normalization:
        x = keras.layers.BatchNormalization()(x)
    
    if dilation > 0:
        x1 = keras.layers.Conv1DTranspose(filters=rev_ae_conv_filter_counts[-1], kernel_size=rev_ae_conv_kernel_sizes[-1], strides=rev_ae_conv_strides[-1], activation="linear", padding="same")(x)
        x2 = keras.layers.Conv1DTranspose(filters=rev_ae_conv_filter_counts[-1], kernel_size=1, strides=stride, activation="linear", padding="same")(x)
        x2 = keras.layers.Conv1DTranspose(filters=rev_ae_conv_filter_counts[-1], kernel_size=rev_ae_conv_kernel_sizes[-1], dilation_rate=rev_ae_conv_dilations[-1], activation="linear", padding="same")(x2)
        x = keras.layers.Add()([x1, x2])
    else:
        x = keras.layers.Conv1DTranspose(filters=rev_ae_conv_filter_counts[-1], kernel_size=rev_ae_conv_kernel_sizes[-1], strides=rev_ae_conv_strides[-1], activation="linear", padding="same")(x)
    x = keras.layers.Reshape([audio_window_length])(x)

ae_decoder_out = x
ae_decoder = keras.Model(inputs=ae_decoder_in, outputs=ae_decoder_out)
ae_decoder.summary()

if save_models == True:
    ae_decoder.save("aae/models/ae_decoder")
    keras.utils.plot_model(ae_decoder, show_shapes=True, dpi=64, to_file='aae/models/ae_decoder.png')

if load_weights and ae_decoder_weights_file:
    ae_decoder.load_weights(ae_decoder_weights_file)

# Create Autoencoder

ae = tf.keras.Model(inputs=ae_encoder_in, outputs=[ae_decoder(ae_encoder_out)])
ae.summary()

if save_models == True:
    ae.save("aae/models/ae")
    keras.utils.plot_model(ae, show_shapes=True, dpi=64, to_file='aae/models/ae.png')

if load_weights and ae_weights_file:
    ae.load_weights(ae_weights_file)

# Loss functions

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

# discriminator prior loss
def disc_prior_loss(d_x, g_z, smoothing_factor = 0.9):
    real_loss = cross_entropy(tf.ones_like(d_x) * smoothing_factor, d_x) # If we feed the discriminator with real images, we assume they all are the right pictures --> Because of that label == 1
    fake_loss = cross_entropy(tf.zeros_like(g_z), g_z) # Each noise we feed in are fakes image --> Because of that labels are 0. 
    total_loss = real_loss + fake_loss
    
    return total_loss

# discriminator audio loss
def disc_audio_loss(d_x, g_z, smoothing_factor = 0.9):
    real_loss = cross_entropy(tf.ones_like(d_x) * smoothing_factor, d_x) # If we feed the discriminator with real images, we assume they all are the right pictures --> Because of that label == 1
    fake_loss = cross_entropy(tf.zeros_like(g_z), g_z) # Each noise we feed in are fakes image --> Because of that labels are 0. 
    total_loss = real_loss + fake_loss
    
    return total_loss

# autoencoder audio reconstrictopm loss
def ae_rec_loss(y, yhat):
    flat_y = tf.keras.backend.flatten(y)
    flat_yhat = tf.keras.backend.flatten(yhat)
    _loss = tf.reduce_mean((flat_y-flat_yhat)**2)
    
    return tf.keras.backend.mean(_loss)

# autoencoder disc prior loss
def ae_disc_prior_loss(dx_of_gx):
    return cross_entropy(tf.ones_like(dx_of_gx), dx_of_gx) 

# autoencoder disc audio loss
def ae_disc_audio_loss(dx_of_gx):
    return cross_entropy(tf.ones_like(dx_of_gx), dx_of_gx) 

# autoencoder l1 loss
def ae_l1_loss(z):
    flat_z = tf.keras.backend.flatten(z)
    _loss = tf.reduce_mean(tf.math.abs(flat_z))
    
    return _loss

# autoencoder l2 loss
def ae_l2_loss(z):
    flat_z = tf.keras.backend.flatten(z)
    _loss = tf.reduce_mean(flat_z**2)
    
    return _loss

# optimizers
ae_optimizer = keras.optimizers.Adam(ae_learning_rate)
disc_prior_optimizer = keras.optimizers.Adam(disc_prior_learning_rate)
disc_audio_optimizer = keras.optimizers.Adam(disc_audio_learning_rate)

# Train Step

@tf.function()
def train_step(ae, disc_prior, disc_audio, audio_batch, batch_size = 32):
    # train disc_prior
    fake_normal = ae_encoder(audio_batch, training=False)
    real_normal = sample_normal(fake_normal.shape)
    
    with tf.GradientTape() as disc_prior_tape:
        disc_prior_real_output =  disc_prior(real_normal, training=True)
        disc_prior_fake_output =  disc_prior(fake_normal, training=True)   
        _disc_prior_loss = disc_prior_loss(disc_prior_real_output, disc_prior_fake_output)

        disc_prior_gradients = disc_prior_tape.gradient(_disc_prior_loss, disc_prior.trainable_variables)
        disc_prior_optimizer.apply_gradients(zip(disc_prior_gradients, disc_prior.trainable_variables))
        
    # train disc_audio
    fake_audio = ae(audio_batch, training=False)
    real_audio = audio_batch
    
    with tf.GradientTape() as disc_audio_tape:
        disc_audio_real_output =  disc_audio(real_audio, training=True)
        disc_audio_fake_output =  disc_audio(fake_audio, training=True)   
        _disc_audio_loss = disc_audio_loss(disc_audio_real_output, disc_audio_fake_output)

        disc_audio_gradients = disc_audio_tape.gradient(_disc_audio_loss, disc_audio.trainable_variables)
        disc_audio_optimizer.apply_gradients(zip(disc_audio_gradients, disc_audio.trainable_variables))

    # train autoencoder
    with tf.GradientTape() as ae_tape:
        encoder_out = ae_encoder(audio_batch, training=True)  
        decoder_out = ae_decoder(encoder_out, training=True)
        
        _ae_rec_loss = ae_rec_loss(audio_batch, decoder_out) 
        _ae_disc_prior_loss = ae_disc_prior_loss(disc_prior_fake_output)
        _ae_disc_audio_loss = ae_disc_audio_loss(disc_audio_fake_output)
        _ae_l1_loss = ae_l1_loss(encoder_out)
        _ae_l2_loss = ae_l2_loss(encoder_out)
        
        _ae_loss = _ae_rec_loss * ae_rec_loss_scale + _ae_disc_prior_loss * ae_disc_prior_loss_scale + _ae_disc_audio_loss * ae_disc_audio_loss_scale + _ae_l1_loss * ae_l1_loss_scale + _ae_l2_loss * ae_l2_loss_scale

        ae_gradients = ae_tape.gradient(_ae_loss, ae.trainable_variables)
        ae_optimizer.apply_gradients(zip(ae_gradients, ae.trainable_variables))
        
        return _ae_loss, _disc_prior_loss, _disc_audio_loss, _ae_rec_loss, _ae_disc_prior_loss, _ae_disc_audio_loss, _ae_l1_loss, _ae_l2_loss
  
# Test Step
      
@tf.function()
def test_step(ae, disc_prior, disc_audio, audio_batch, batch_size = 32):
    # test disc_prior
    fake_normal = ae_encoder(audio_batch, training=False)
    real_normal = sample_normal(fake_normal.shape)
    
    disc_prior_real_output =  disc_prior(real_normal, training=False)
    disc_prior_fake_output =  disc_prior(fake_normal, training=False)   
    _disc_prior_loss = disc_prior_loss(disc_prior_real_output, disc_prior_fake_output)

    # test disc_audio
    fake_audio = ae(audio_batch, training=False)
    real_audio = audio_batch
    
    disc_audio_real_output =  disc_audio(real_audio, training=False)
    disc_audio_fake_output =  disc_audio(fake_audio, training=False)   
    _disc_audio_loss = disc_audio_loss(disc_audio_real_output, disc_audio_fake_output)

    # test autoencoder
    encoder_out = ae_encoder(audio_batch, training=False)  
    decoder_out = ae_decoder(encoder_out, training=False)
    
    _ae_rec_loss = ae_rec_loss(audio_batch, decoder_out) 
    _ae_disc_prior_loss = ae_disc_prior_loss(disc_prior_fake_output)
    _ae_disc_audio_loss = ae_disc_audio_loss(disc_audio_fake_output)
    _ae_l1_loss = ae_l1_loss(encoder_out)
    _ae_l2_loss = ae_l2_loss(encoder_out)
    
    _ae_loss = _ae_rec_loss * ae_rec_loss_scale + _ae_disc_prior_loss * ae_disc_prior_loss_scale + _ae_disc_audio_loss * ae_disc_audio_loss_scale + _ae_l1_loss * ae_l1_loss_scale + _ae_l2_loss * ae_l2_loss_scale
    return _ae_loss, _disc_prior_loss, _disc_audio_loss, _ae_rec_loss, _ae_disc_prior_loss, _ae_disc_audio_loss, _ae_l1_loss, _ae_l2_loss

# Training Loop

def train(train_dataset, test_dataset, epoches):
    
    loss_history = {}
    loss_history["ae train_loss"] = []
    loss_history["disc prior train loss"] = []
    loss_history["disc audio train loss"] = []
    loss_history["ae rec train loss"] = []
    loss_history["ae l1 train loss"] = []
    loss_history["ae l2 train loss"] = []
    loss_history["ae test loss"] = []
    loss_history["disc prior test loss"] = []
    loss_history["disc audio test loss"] = []
    loss_history["ae rec test loss"] = []
    loss_history["ae l1 test loss"] = []
    loss_history["ae l2 test loss"] = []

    for epoch in range(epoches):
        
        start = time.time()
        
        ae_train_loss_per_epoch = []
        disc_prior_train_loss_per_epoch = []
        disc_audio_train_loss_per_epoch = []
        ae_rec_train_loss_per_epoch = []
        ae_l1_train_loss_per_epoch = []
        ae_l2_train_loss_per_epoch = []
        
        for train_batch in train_dataset:
            
            _ae_loss, _disc_prior_loss, _disc_audio_loss, _ae_rec_loss, _ae_disc_prior_loss, _ae_disc_audio_loss, _ae_l1_loss, _ae_l2_loss = train_step(ae, disc_prior, disc_audio, train_batch, batch_size = batch_size)

            _ae_loss = np.array(_ae_loss)
            _disc_prior_loss = np.array(_disc_prior_loss)
            _disc_audio_loss = np.array(_disc_audio_loss)         
            _ae_rec_loss = np.array(_ae_rec_loss)
            _ae_l1_loss = np.array(_ae_l1_loss)
            _ae_l2_loss = np.array(_ae_l2_loss)

            ae_train_loss_per_epoch.append(_ae_loss)
            disc_prior_train_loss_per_epoch.append(_disc_prior_loss)
            disc_audio_train_loss_per_epoch.append(_disc_audio_loss)
            ae_rec_train_loss_per_epoch.append(_ae_rec_loss)
            ae_l1_train_loss_per_epoch.append(_ae_l1_loss)
            ae_l2_train_loss_per_epoch.append(_ae_l2_loss)
        
        ae_train_loss_per_epoch = np.mean(np.array(ae_train_loss_per_epoch))
        disc_prior_train_loss_per_epoch = np.mean(np.array(disc_prior_train_loss_per_epoch))
        disc_audio_train_loss_per_epoch = np.mean(np.array(disc_audio_train_loss_per_epoch))
        ae_rec_train_loss_per_epoch = np.mean(np.array(ae_rec_train_loss_per_epoch))
        ae_l1_train_loss_per_epoch = np.mean(np.array(ae_l1_train_loss_per_epoch))
        ae_l2_train_loss_per_epoch = np.mean(np.array(ae_l2_train_loss_per_epoch))
        
        ae_test_loss_per_epoch = []
        disc_prior_test_loss_per_epoch = []
        disc_audio_test_loss_per_epoch = []
        ae_rec_test_loss_per_epoch = []
        ae_l1_test_loss_per_epoch = []
        ae_l2_test_loss_per_epoch = []
        
        for test_batch in test_dataset:
            _ae_loss, _disc_prior_loss, _disc_audio_loss, _ae_rec_loss, _ae_disc_prior_loss, _ae_disc_audio_loss, _ae_l1_loss, _ae_l2_loss = test_step(ae, disc_prior, disc_audio, test_batch, batch_size = batch_size)
            
            _ae_loss = np.array(_ae_loss)
            _disc_prior_loss = np.array(_disc_prior_loss)
            _disc_audio_loss = np.array(_disc_audio_loss)         
            _ae_rec_loss = np.array(_ae_rec_loss)
            _ae_l1_loss = np.array(_ae_l1_loss)
            _ae_l2_loss = np.array(_ae_l2_loss)

            ae_test_loss_per_epoch.append(_ae_loss)
            disc_prior_test_loss_per_epoch.append(_disc_prior_loss)
            disc_audio_test_loss_per_epoch.append(_disc_audio_loss)
            ae_rec_test_loss_per_epoch.append(_ae_rec_loss)
            ae_l1_test_loss_per_epoch.append(_ae_l1_loss)
            ae_l2_test_loss_per_epoch.append(_ae_l2_loss)
            
            
        ae_test_loss_per_epoch = np.mean(np.array(ae_test_loss_per_epoch))
        disc_prior_test_loss_per_epoch = np.mean(np.array(disc_prior_test_loss_per_epoch))
        disc_audio_test_loss_per_epoch = np.mean(np.array(disc_audio_test_loss_per_epoch))
        ae_rec_test_loss_per_epoch = np.mean(np.array(ae_rec_test_loss_per_epoch))
        ae_l1_test_loss_per_epoch = np.mean(np.array(ae_l1_test_loss_per_epoch))
        ae_l2_test_loss_per_epoch = np.mean(np.array(ae_l2_test_loss_per_epoch))

        if epoch % model_save_interval == 0 and save_weights == True:
            disc_prior.save_weights("aae/weights/disc_prior_weights epoch_{}".format(epoch))
            disc_audio.save_weights("aae/weights/disc_audio_weights epoch_{}".format(epoch))
            ae_encoder.save_weights("aae/weights/ae_encoder_weights epoch_{}".format(epoch))
            ae_decoder.save_weights("aae/weights/ae_decoder_weights epoch_{}".format(epoch))
            ae.save_weights("aae/weights/ae_weights epoch_{}".format(epoch))
        
        if epoch % audio_save_interval == 0 and save_audio == True:
            create_epoch_sonifications(epoch)
            
        loss_history["ae train_loss"].append(ae_train_loss_per_epoch)
        loss_history["disc prior train loss"].append(disc_prior_train_loss_per_epoch)
        loss_history["disc audio train loss"].append(disc_audio_train_loss_per_epoch)
        loss_history["ae rec train loss"].append(ae_rec_train_loss_per_epoch)
        loss_history["ae l1 train loss"].append(ae_l1_train_loss_per_epoch)
        loss_history["ae l2 train loss"].append(ae_l2_train_loss_per_epoch)
        loss_history["ae test loss"].append(ae_test_loss_per_epoch)
        loss_history["disc prior test loss"].append(disc_prior_test_loss_per_epoch)
        loss_history["disc audio test loss"].append(disc_audio_test_loss_per_epoch)
        loss_history["ae rec test loss"].append(ae_rec_test_loss_per_epoch)
        loss_history["ae l1 test loss"].append(ae_l1_test_loss_per_epoch)
        loss_history["ae l2 test loss"].append(ae_l2_test_loss_per_epoch)

        print ('epoch {} :  ae train {:01.4f} dprior {:01.4f} daudio {:01.4f} rec {:01.4f} l1 {:01.4f} l2 {:01.4f} test {:01.4f} dprior {:01.4f} daudio {:01.4f} rec {:01.4f} l1 {:01.4f} l2 {:01.4f} time {:01.2f}'.format(epoch + 1, ae_train_loss_per_epoch, disc_prior_train_loss_per_epoch, disc_audio_train_loss_per_epoch, 
               ae_rec_train_loss_per_epoch, ae_l1_train_loss_per_epoch, ae_l2_train_loss_per_epoch, ae_test_loss_per_epoch, disc_prior_test_loss_per_epoch, disc_audio_test_loss_per_epoch, 
               ae_rec_test_loss_per_epoch, ae_l1_test_loss_per_epoch, ae_l2_test_loss_per_epoch, time.time()-start))
    
    return loss_history

# Save Reconstructd Audio Examples
def create_pred_sonification(start_time, duration, file_name):

    grain_env = np.hanning(audio_window_length)
    grain_offset = audio_sample_rate // audio_window_rate
    
    predict_start_window = int(start_time * audio_save_sample_rate // audio_window_length)
    predict_window_count = int(duration * audio_save_sample_rate // grain_offset)

    pred_audio_sequence_length = (predict_window_count - 1) * grain_offset + audio_window_length
    pred_audio_sequence = np.zeros(shape=(pred_audio_sequence_length), dtype=np.float32)

    for i in range(predict_window_count):
        target_audio = audio_standardized[predict_start_window * audio_window_length + i*grain_offset:predict_start_window * audio_window_length + i*grain_offset + audio_window_length]
        enc_audio = ae_encoder.predict(tf.expand_dims(target_audio, 0))
        pred_audio = ae_decoder.predict(enc_audio)
        pred_audio = np.squeeze(pred_audio)
    
        pred_audio = pred_audio * grain_env

        pred_audio_sequence[i*grain_offset:i*grain_offset + audio_window_length] = pred_audio_sequence[i*grain_offset:i*grain_offset + audio_window_length] + pred_audio


    if standardize_audio:
        pred_audio_sequence = audio_mean + pred_audio_sequence * audio_std

    # cpnvert audio float array to audio int16 array with samples between -2**15 amd 2**15
    pred_audio_int16_sequence = np.clip(pred_audio_sequence, -1.0, 1.0)
    pred_audio_int16_sequence = pred_audio_int16_sequence * audio_sample_scale
    pred_audio_int16_sequence = pred_audio_int16_sequence.astype(np.int16)
    
    audio_write(file_name, audio_save_sample_rate, pred_audio_int16_sequence)

# Save Interpolated Audio Examples
def create_interpol_sonification(start_time1, start_time2, duration, file_name):
    grain_env = np.hanning(audio_window_length)
    grain_offset = audio_sample_rate // audio_window_rate

    start_alpha = -1.0
    end_alpha = 2.0

    audio_window_count = int(duration * audio_window_rate)
    pred_audio_sequence_length = int((audio_window_count - 1) * grain_offset + audio_window_length)
    pred_audio_sequence = np.zeros(shape=(pred_audio_sequence_length), dtype=np.float32)

    for i in range(audio_window_count):
        
        target1_audio_start = int(start_time1 * audio_save_sample_rate + i * grain_offset)
        target1_audio_end = target1_audio_start + audio_window_length
        
        target2_audio_start = int(start_time2 * audio_save_sample_rate + i * grain_offset)
        target2_audio_end = target2_audio_start + audio_window_length
    
        seq1_audio_window = audio_standardized[target1_audio_start:target1_audio_end]
        seq1_audio_window = tf.expand_dims(seq1_audio_window, 0)  
        seq1_latent_vector = ae_encoder(seq1_audio_window, training=False)
    
        seq2_audio_window = audio_standardized[target2_audio_start:target2_audio_end]
        seq2_audio_window = tf.expand_dims(seq2_audio_window, 0)  
        seq2_latent_vector = ae_encoder(seq2_audio_window, training=False)
        
        alpha = start_alpha + (end_alpha - start_alpha) * i / (audio_window_count - 1)
        mix_latent_vector = seq1_latent_vector * (1.0 - alpha) + seq2_latent_vector * alpha
        
        pred_audio = ae_decoder(mix_latent_vector, training=False)
        
        pred_audio *= grain_env
        
        pred_audio_start = int(i * grain_offset)
        pred_audio_end = pred_audio_start + audio_window_length

        pred_audio_sequence[pred_audio_start:pred_audio_end] += pred_audio
    
    if standardize_audio:
        pred_audio_sequence = audio_mean + pred_audio_sequence * audio_std

    # convert audio float array to audio int16 array with samples between -2**15 amd 2**15
    pred_audio_int16_sequence = np.clip(pred_audio_sequence, -1.0, 1.0)
    pred_audio_int16_sequence = pred_audio_int16_sequence * audio_sample_scale
    pred_audio_int16_sequence = pred_audio_int16_sequence.astype(np.int16)

    audio_write(file_name, audio_save_sample_rate, pred_audio_int16_sequence)

# Save Latent Space Traversal Audio Examples
def create_traverse_sonifcation(start_audio_window_index, end_audio_window_index, window_count, window_interpolation_count, file_name):
    # interpolate audio latent space
    grain_env = np.hanning(audio_window_length)
    grain_offset = audio_sample_rate // audio_window_rate

    target_windows = list(np.linspace(start_audio_window_index, end_audio_window_index, window_count, dtype=np.int32))
    interpolation_window_count = window_interpolation_count
    target_audio = []

    audio_window_count = len(target_windows)
    
    pred_audio_sequence_length = int((window_interpolation_count * audio_window_count - 1) * grain_offset + audio_window_length)
    pred_audio_sequence = np.zeros(shape=(pred_audio_sequence_length), dtype=np.float32)

    for wI in range(audio_window_count - 1):
    
        target_start_window = target_windows[wI]
        target_end_window = target_windows[wI + 1]
    
        target_start_audio = np.expand_dims(audio_standardized[target_start_window*audio_window_length:(target_start_window + 1) * audio_window_length], axis=0)
        target_end_audio = np.expand_dims(audio_standardized[target_end_window*audio_window_length:(target_end_window + 1) * audio_window_length], axis=0)

        start_enc = ae_encoder(target_start_audio)
        end_enc = ae_encoder(target_end_audio)

        for i in range(interpolation_window_count):
            inter_enc = start_enc + (end_enc - start_enc) * i / (interpolation_window_count - 1.0)
            
            pred_audio = ae_decoder(inter_enc, training=False)
            
            pred_audio *= grain_env

            i2 = (wI * interpolation_window_count + i)
            
            pred_audio_start = int(i2 * grain_offset)
            pred_audio_end = pred_audio_start + audio_window_length
            
            pred_audio_sequence[pred_audio_start:pred_audio_end] += pred_audio
        
            #print("predict window ", wI, " step ", i)
        
    if standardize_audio:
        pred_audio_sequence = audio_mean + pred_audio_sequence * audio_std

    # cpnvert audio float array to audio int16 array with samples between -2**15 amd 2**15
    pred_audio_int16_sequence = np.clip(pred_audio_sequence, -1.0, 1.0)
    pred_audio_int16_sequence = pred_audio_int16_sequence * audio_sample_scale
    pred_audio_int16_sequence = pred_audio_int16_sequence.astype(np.int16)

    #play_obj = sa.play_buffer(pred_audio_int16_sequence, 1, 2, audio_save_sample_rate)
    #play_obj.stop()
    
    audio_write(file_name, audio_save_sample_rate, pred_audio_int16_sequence)

# Create Audio Examples for Each Epoch
def create_epoch_sonifications(epoch):
    # pred sonifications
    for audio_start_time in audio_save_start_times:
        file_name = "../results/aae/pred_audio_{}_epoch_{}.wav".format(audio_start_time, epoch)
        create_pred_sonification(audio_start_time, audio_save_duration, file_name)

    # two audio region interpolation
    for time_index in range(len(audio_save_start_times) - 1):
        audio_start_time1 = audio_save_start_times[time_index]
        audio_start_time2 = audio_save_start_times[time_index + 1]
        file_name = "../results/aae/interpol_audio_{}_{}_epoch_{}.wav".format(audio_start_time1, audio_start_time2, epoch)
        create_interpol_sonification(audio_start_time1, audio_start_time2, audio_save_duration, file_name)

    # audio window sequence traversal
    file_name = "../results/aae/traverse_audio_{}_{}_{}_{}_epoch_{}.wav".format(audio_traverse_start_window, audio_traverse_end_window, audio_traverse_window_count, audio_traverse_interpolation_count, epoch)
    create_traverse_sonifcation(audio_traverse_start_window, audio_traverse_end_window, audio_traverse_window_count, audio_traverse_interpolation_count, file_name)

 
# Train Model
loss_history = train(train_dataset, test_dataset, epochs)

# save history
if save_history:
    utils.save_loss_as_csv(loss_history, "../results/aae/history.csv")
    utils.save_loss_as_image(loss_history, "../results/aae/history.png")

# final weights save
if save_weights:
    disc_prior.save_weights("aae/weights/disc_prior_weights_epoch_{}".format(epochs))
    disc_audio.save_weights("aae/weights/disc_audio_weights_epoch_{}".format(epochs))
    ae_encoder.save_weights("aae/weights/ae_encoder_weights_epoch_{}".format(epochs))
    ae_decoder.save_weights("aae/weights/ae_decoder_weights_epoch_{}".format(epochs))
    ae.save_weights("aae/weights/ae_weights_epoch_{}".format(epochs))

if save_audio:
    create_epoch_sonifications(epochs)
