"""
seq2seq model translating mocap pose sequences into encoded audio window sequences
this model doesn't employ attention

Notes on discriminators:
the model uses two discriminators, one for distinguishing between real and fake audio encodings
and one for distinguishing between real and fake audio. The latter is identical to the one used
for the adversarial autoencoder. 
"""

# Imports

import tensorflow as tf
import tensorflow.keras as keras
import simpleaudio as sa
from scipy.io.wavfile import read as audio_read
from scipy.io.wavfile import write as audio_write
import math
import numpy as np
import pickle      
import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation 
import mpl_toolkits.mplot3d as plt3d
import os, time
from common.pose_renderer import PoseRenderer
from common import utils  
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.pose_anim_player import PoseAnimPlayer
from common.quaternion import qmul, qnormalize_np, slerp

# Mocap Configuration

mocap_data_path = "../data/improvisation_mocap.p"
pose_fps=50
pose_window_length = 8
pose_sequence_latent_dimension = 128
standardize_poses = True
mocap_valid_frame_ranges = [ [ 1500, 3450 ],
                            [ 3510, 3570 ],
                            [ 3625, 13140 ],
                            [ 13180, 16370 ],
                            [ 16420, 20745 ],
                            [ 20770, 25390 ],
                            [ 25445, 29180] ]

mocap_valid_frame_count = np.sum([ range[1] - range[0] for range in mocap_valid_frame_ranges ])

# Audio Configuration
#audio_file_path = "../data/sonification_audio"
audio_file_path = "../data/improvisation_audio"
audio_save_sample_rate = 32000 # sample rate for saving example sonifications
audio_sample_rate = 32000 # numer of audio samples per sec
audio_window_length = 256 # size of audio window in number of samples (must match saved audio model)
audio_window_rate = 250 # number of audio windows per sec (must match saved audio model)
standardize_audio = True
audio_latent_dimension = 32 # this dimension must match what the preloaded audio models (audio_encoder, audio_decoder, audio_disc) have been trained with
audio_window_offset = audio_sample_rate // audio_window_rate
audio_samples_per_pose = audio_sample_rate // pose_fps
audio_samples_per_pose_window = pose_window_length * audio_samples_per_pose
audio_windows_per_pose_window = (audio_samples_per_pose_window - audio_window_length) // audio_window_offset + 1

# Autoencoder Models and Weights
audio_encoder_model_file_path = "aae/models/improvisation/ae_encoder" 
audio_decoder_model_file_path = "aae/models/improvisation/ae_decoder" 
audio_disc_model_file_path = "aae/models/improvisation/disc_audio" 
audio_encoder_weights_file_path = "aae/weights/improvisation/ae_encoder_weights_epoch_300" 
audio_decoder_weights_file_path = "aae/weights/improvisation/ae_decoder_weights_epoch_300" 
audio_disc_weights_file_path = "aae/weights/improvisation/disc_audio_weights_epoch_300" 

# Model Configuration

# Seq2Seq Configuration
seq2seq_rnn_layer_sizes = [512, 512, 512]
final_decoder_activation = "linear"

# L^atent Audio Discriminator
laudio_disc_dense_layer_sizes = [ 32, 32 ]

# Save / Load Preprocesed Mocap and Audio Data
load_data = False
save_data = True
data_file_path = "../data/mocap_laudio_improvisation_data"

# Save / Load Model Weights
save_models = False
load_weights = True
seq2seq_encoder_weights_file_path = "seq2seq/weights/improvisation/seq2seq_encoder_epoch_300"
seq2seq_decoder_weights_file_path = "seq2seq/weights/improvisation/seq2seq_decoder_epoch_300"
laudio_disc_weights_file_path = "seq2seq/weights/improvisation/laudio_disc_epoch_300"

# Training Configuration
pose_sequence_offset = 1
batch_size = 16
train_percentage = 0.8 # train / test split
test_percentage  = 0.2
epochs = 200

laudio_rec_loss_scale = 1.0
audio_rec_loss_scale = 0.1
laudio_disc_loss_scale = 0.1
audio_disc_loss_scale = 0.1
laudio_l2_loss_scale = 0.01 # l2 loss on predicted audio window encodings
audio_l2_loss_scale = 0.0 # l2 loss on the predicted audio windows
state_l2_loss_scale = 0.0 # l2 loss on encoder state vectors
seq2seq_learning_rate = 1e-4
laudio_disc_learning_rate = 5e-4

# Mocap Visualisation Settings
save_vis = False
view_ele = 90.0
view_azi = -90.0
view_line_width = 4.0
view_size = 8.0
vis_save_interval = 100
vis_pose_frames = [0, 1000, 2000]

# Load Mocap Data
mocap_data = MocapDataset(mocap_data_path, fps=pose_fps)
mocap_data.compute_positions()

skeleton = mocap_data.skeleton()
skeleton_joint_count = skeleton.num_joints()
skel_edge_list = utils.get_skeleton_edge_list(skeleton)
poseRenderer = PoseRenderer(skel_edge_list)

subject = np.random.choice(list(mocap_data.subjects()))
action = np.random.choice(list(mocap_data.subject_actions(subject)))
pose_sequence = mocap_data[subject][action]["rotations"]
total_sequence_length = pose_sequence.shape[0]
joint_count = pose_sequence.shape[1]
joint_dim = pose_sequence.shape[2]
pose_dim = joint_count * joint_dim
pose_sequence = np.reshape(pose_sequence, (-1, pose_dim))

if standardize_poses:
    pose_mean = np.mean(pose_sequence, axis=0)
    pose_std = np.std(pose_sequence, axis=0)
    pose_standardized = (pose_sequence - pose_mean) / (pose_std + 0.00001)
else:
    pose_standardized = pose_sequence   


# Load Audio Data
audio = audio_read(audio_file_path + ".wav")
audio_sample_scale = 2**15
audio_float_array = np.array(audio[1], dtype=np.float32)
audio_float_array = audio_float_array / audio_sample_scale

if standardize_audio:
    audio_mean = np.mean(audio_float_array)
    audio_std = np.std(audio_float_array)
    audio_standardized = (audio_float_array - audio_mean) / (audio_std)
else:
    audio_standardized = audio_float_array
    
# Load Autencoder Models and Weights: Audio Encoder, Audio Decoder and Audio Discriminator
audio_encoder = tf.keras.models.load_model(audio_encoder_model_file_path)
audio_decoder = tf.keras.models.load_model(audio_decoder_model_file_path)
audio_disc = tf.keras.models.load_model(audio_disc_model_file_path)

audio_encoder.load_weights(audio_encoder_weights_file_path)
audio_decoder.load_weights(audio_decoder_weights_file_path)
audio_disc.load_weights(audio_disc_weights_file_path)

# Save / Load Preprocesed Mocap and Audio Data
if load_data and data_file_path:
    data = pickle.load(open(data_file_path + ".p", "rb"))
    pose_windows = data["pose_windows"]
    audio_window_sequences = data["audio_window_sequences"]
    audio_seq_encodings = data["audio_seq_encodings"]
else:
    # collect matching pose and audio sequence excerpts
    pose_windows = []
    audio_window_sequences = []
    
    for pose_valid_frame_range in mocap_valid_frame_ranges:
        
        pose_frame_range_start = pose_valid_frame_range[0]
        pose_frame_range_end = pose_valid_frame_range[1]
        
        for pose_window_start_frame in np.arange(pose_frame_range_start, pose_frame_range_end - pose_window_length, pose_sequence_offset):
            
            #print("valid: start ", frame_range_start, " end ", frame_range_end, " exc: start ", pose_seq_excerpt_start, " end ", (pose_seq_excerpt_start + pose_window_length) )
            
            pose_window = pose_standardized[pose_window_start_frame:pose_window_start_frame + pose_window_length]
            pose_windows.append(pose_window)
            
            audio_windows = []
    
            for audio_window_index in range(-1, audio_windows_per_pose_window):
                audio_window_start_sample = pose_window_start_frame * audio_samples_per_pose + audio_window_index * audio_window_offset
                audio_window_end_sample = audio_window_start_sample + audio_window_length
                
                #print("pos ", pose_window_start_frame, " awi ", audio_window_index , " aws ", audio_window_start_sample, " awe ", audio_window_end_sample)
                
                audio_window =  audio_standardized[audio_window_start_sample:audio_window_end_sample]
                audio_windows.append(audio_window)
                
                #print("audio_window s ", audio_window.shape)
            
            audio_windows = np.array(audio_windows)
            
            #print("audio_windows s ", audio_windows.shape)
            
            audio_window_sequences.append(audio_windows)
    
    sequence_count = len(pose_windows)
    
    pose_windows = np.array(pose_windows)
    audio_window_sequences = np.array(audio_window_sequences)
    
    # create matching audio encodings
    audio_seq_encodings = []
    
    for seq_index in range(sequence_count):
    
        print("seq ", seq_index, " out of ", sequence_count)
        
        audio_windows = audio_window_sequences[seq_index]
        
        audio_window_encodings = []
        for audio_window_index in range(audio_windows_per_pose_window + 1):
            
            audio_window = audio_windows[audio_window_index]
            audio_window_encoding = audio_encoder(np.expand_dims(audio_window, axis=0), training=False)
            audio_window_encoding = np.squeeze(audio_window_encoding, axis=0)
            
            #print("sI ", seq_index, " pI ", pose_index ," ase ", audio_subseq_encoding.shape)
            
            audio_window_encodings.append(audio_window_encoding)
        
        audio_window_encodings = np.array(audio_window_encodings)
        
        #print("sI ", seq_index, " ase2 ", audio_subseq_encodings.shape)
        
        audio_seq_encodings.append(audio_window_encodings)
    
    audio_seq_encodings = np.array(audio_seq_encodings)

    if save_data and data_file_path:
        data = {}
        data["pose_windows"] = pose_windows
        data["audio_window_sequences"] = audio_window_sequences
        data["audio_seq_encodings"] = audio_seq_encodings
        
        pickle.dump(data, open(data_file_path + ".p", "wb"))

sequence_count = pose_windows.shape[0]

# Dreate Dataset

dataset = tf.data.Dataset.from_tensor_slices((pose_windows, audio_window_sequences, audio_seq_encodings))
dataset = dataset.shuffle(sequence_count).batch(batch_size, drop_remainder=True)

dataset_size = sequence_count // batch_size

train_size = int(train_percentage * dataset_size)
test_size = int(test_percentage * dataset_size)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Create Models

# Seq2Seq Encoder
seq2seq_encoder_in = keras.layers.Input(shape=(None, pose_dim))
seq2seq_encoder_out_states = []

x = seq2seq_encoder_in

for layer_size in seq2seq_rnn_layer_sizes:
    x, state_h, state_c = keras.layers.LSTM(layer_size, activation="tanh", return_sequences=True, return_state=True)(x)  
    seq2seq_encoder_out_states.append(state_h)
    seq2seq_encoder_out_states.append(state_c)

seq2seq_encoder = keras.Model(seq2seq_encoder_in, seq2seq_encoder_out_states)
seq2seq_encoder.summary()

if save_models == True:
    seq2seq_encoder.save("seq2seq/models/seq2seq_encoder")
    keras.utils.plot_model(seq2seq_encoder, show_shapes=True, dpi=64, to_file='seq2seq/models/seq2seq_encoder.png')

if load_weights and seq2seq_encoder_weights_file_path:
    seq2seq_encoder.load_weights(seq2seq_encoder_weights_file_path)

# Seq2Seq Decoder
seq2seq_decoder_in = keras.layers.Input(shape=(None, audio_latent_dimension))
seq2seq_decoder_in_states = []
seq2seq_decoder_out_states = []

x = seq2seq_decoder_in

for layer_size in seq2seq_rnn_layer_sizes:
    in_state_h = keras.layers.Input(layer_size)
    in_state_c = keras.layers.Input(layer_size)
    x, out_state_h, out_state_c = keras.layers.LSTM(layer_size, activation="tanh", return_sequences=True, return_state=True)(x, initial_state=[in_state_h, in_state_c])
    
    seq2seq_decoder_in_states.append(in_state_h)
    seq2seq_decoder_in_states.append(in_state_c)
    seq2seq_decoder_out_states.append(out_state_h)
    seq2seq_decoder_out_states.append(out_state_c)

x = keras.layers.Dense(audio_latent_dimension, activation=final_decoder_activation)(x)
seq2seq_decoder_out = x

seq2seq_decoder = keras.Model(inputs=[seq2seq_decoder_in] + seq2seq_decoder_in_states, outputs=[seq2seq_decoder_out] + seq2seq_decoder_out_states)
seq2seq_decoder.summary()

if save_models == True:
    seq2seq_decoder.save("seq2seq/models/seq2seq_decoder")
    keras.utils.plot_model(seq2seq_decoder, show_shapes=True, dpi=64, to_file='seq2seq/models/seq2seq_decoder.png')

if load_weights and seq2seq_decoder_weights_file_path:
    seq2seq_decoder.load_weights(seq2seq_decoder_weights_file_path)
    
# Latent Audio Discriminator

laudio_disc_in = keras.layers.Input(shape=(audio_windows_per_pose_window, audio_latent_dimension))
x = laudio_disc_in

for layer_size in laudio_disc_dense_layer_sizes:
    x = keras.layers.Dense(layer_size)(x)
    x = keras.layers.ELU()(x)    
x = keras.layers.Dense(1, activation="sigmoid")(x)

laudio_disc_out = x
laudio_disc = keras.Model(laudio_disc_in, laudio_disc_out)
laudio_disc.summary()

if save_models == True:
    laudio_disc.save("seq2seq/models/laudio_disc")
    keras.utils.plot_model(laudio_disc, show_shapes=True, dpi=64, to_file='seq2seq/models/laudio_disc.png')

if load_weights and laudio_disc_weights_file_path:
    laudio_disc.load_weights(laudio_disc_weights_file_path)


# Training

seq2seq_optimizer = tf.keras.optimizers.Adam(seq2seq_learning_rate)
laudio_disc_optimizer = tf.keras.optimizers.Adam(laudio_disc_learning_rate)

# Loss Functions

# latent audio discriminator loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def disc_loss(d_x, g_z, smoothing_factor = 0.9):
    real_loss = cross_entropy(tf.ones_like(d_x) * smoothing_factor, d_x) # If we feed the discriminator with real images, we assume they all are the right pictures --> Because of that label == 1
    fake_loss = cross_entropy(tf.zeros_like(g_z), g_z) # Each noise we feed in are fakes image --> Because of that labels are 0. 
    total_loss = real_loss + fake_loss
    
    return total_loss

def disc_gen_loss(dx_of_gx):
    return cross_entropy(tf.ones_like(dx_of_gx), dx_of_gx) 

def l2_loss(z):
    flat_z = tf.keras.backend.flatten(z)
    _loss = tf.reduce_mean(flat_z**2)
    
    return _loss

def rec_loss(y, yhat):
    flat_y = keras.backend.flatten(y)
    flat_yhat = keras.backend.flatten(yhat)
    rec_loss = tf.reduce_mean((flat_y-flat_yhat)**2)
    return keras.backend.mean(rec_loss)

# Training and Testing Functions

@tf.function
def laudio_disc_train_step(pose_batch, laudio_batch):

    real_laudio = laudio_batch[:, 1:, :]
    
    encoder_states = seq2seq_encoder(pose_batch, training=False)
    decoder_out = seq2seq_decoder( [laudio_batch[:, :-1, :]] + encoder_states, training=False)

    fake_laudio = decoder_out[0]
    
    with tf.GradientTape() as laudio_disc_tape:
        laudio_disc_real_output =  laudio_disc(real_laudio, training=True)
        laudio_disc_fake_output =  laudio_disc(fake_laudio, training=True)

        _loss = disc_loss(laudio_disc_real_output, laudio_disc_fake_output)

        laudio_disc_gradients = laudio_disc_tape.gradient(_loss, laudio_disc.trainable_variables)
        laudio_disc_optimizer.apply_gradients(zip(laudio_disc_gradients, laudio_disc.trainable_variables))  
    
    return _loss

@tf.function
def laudio_disc_test_step(pose_batch, laudio_batch):
    real_laudio = laudio_batch[:, 1:, :]
    
    encoder_states = seq2seq_encoder(pose_batch, training=False)
    decoder_out = seq2seq_decoder( [laudio_batch[:, :-1, :]] + encoder_states, training=False)
    fake_laudio = decoder_out[0]    

    laudio_disc_real_output =  laudio_disc(real_laudio, training=False)
    laudio_disc_fake_output =  laudio_disc(fake_laudio, training=False)
        
    _loss = disc_loss(laudio_disc_real_output, laudio_disc_fake_output)

    return _loss

@tf.function
def laudio_encoder_train_step(pose_batch, audio_batch, laudio_batch):
    real_laudio = laudio_batch
    real_audio = audio_batch
        
    with tf.GradientTape() as tape:
        
        encoder_states = seq2seq_encoder(pose_batch, training=True)
        decoder_out = seq2seq_decoder([ laudio_batch[:,:-1,:] ] + encoder_states, training=True)
        pred_laudio = decoder_out[0]
        pred_audio = audio_decoder(tf.reshape(pred_laudio, shape=[-1, audio_latent_dimension]), training=False)
        pred_audio = tf.reshape(pred_audio, shape=[-1, audio_windows_per_pose_window, audio_window_length])
        
        # laudio disc inference
        laudio_disc_fake_output = laudio_disc(pred_laudio, training=False)
        
        # audio disc inference
        audio_disc_fake_output = audio_disc(tf.reshape(pred_audio, shape=[-1, audio_window_length]), training=False)
        
        _laudio_rec_loss = rec_loss(laudio_batch[:, 1:, :], pred_laudio)
        _audio_rec_loss = rec_loss(audio_batch[:, 1:, :], pred_audio)
        _laudio_disc_loss = disc_gen_loss(laudio_disc_fake_output)
        _audio_disc_loss = disc_gen_loss(audio_disc_fake_output)
        _laudio_l2_loss = l2_loss(pred_laudio)
        _audio_l2_loss = l2_loss(pred_audio)
        _state_l2_loss = l2_loss(tf.concat(values=encoder_states, axis=1))
                
        _loss = 0.0
        _loss += _laudio_rec_loss * laudio_rec_loss_scale
        _loss += _audio_rec_loss * audio_rec_loss_scale
        _loss += _laudio_disc_loss * laudio_disc_loss_scale
        _loss += _audio_disc_loss * audio_disc_loss_scale
        _loss += _laudio_l2_loss * laudio_l2_loss_scale
        _loss += _audio_l2_loss * audio_l2_loss_scale
        _loss += _state_l2_loss * state_l2_loss_scale

        trainable_variables = seq2seq_encoder.trainable_variables + seq2seq_decoder.trainable_variables
        
        gradients = tape.gradient(_loss, trainable_variables)
        seq2seq_optimizer.apply_gradients(zip(gradients, trainable_variables))
    
    return _loss, _laudio_rec_loss, _audio_rec_loss, _laudio_disc_loss, _audio_disc_loss, _laudio_l2_loss, _audio_l2_loss, _state_l2_loss

@tf.function
def laudio_encoder_test_step(pose_batch, audio_batch, laudio_batch):
    real_laudio = laudio_batch
    real_audio = audio_batch

    encoder_states = seq2seq_encoder(pose_batch, training=False)
    decoder_in = [ laudio_batch[:,:-1,:] ] + encoder_states
    decoder_out = seq2seq_decoder(decoder_in, training=False)
    pred_laudio = decoder_out[0]
    pred_audio = audio_decoder(tf.reshape(pred_laudio, shape=[-1, audio_latent_dimension]), training=False)
    pred_audio = tf.reshape(pred_audio, shape=[-1, audio_windows_per_pose_window, audio_window_length])
    
    # laudio disc inference
    laudio_disc_fake_output = laudio_disc(pred_laudio, training=False)
    
    # audio disc inference
    audio_disc_fake_output = audio_disc(tf.reshape(pred_audio, shape=[-1, audio_window_length]), training=False)
    
    _laudio_rec_loss = rec_loss(laudio_batch[:, 1:, :], pred_laudio)
    _audio_rec_loss = rec_loss(audio_batch[:, 1:, :], pred_audio)
    _laudio_disc_loss = disc_gen_loss(laudio_disc_fake_output)
    _audio_disc_loss = disc_gen_loss(audio_disc_fake_output)
    _laudio_l2_loss = l2_loss(pred_laudio)
    _audio_l2_loss = l2_loss(pred_audio)
    _state_l2_loss = l2_loss(tf.concat(values=encoder_states, axis=1))
            
    _loss = 0.0
    _loss += _laudio_rec_loss * laudio_rec_loss_scale
    _loss += _audio_rec_loss * audio_rec_loss_scale
    _loss += _laudio_disc_loss * laudio_disc_loss_scale
    _loss += _audio_disc_loss * audio_disc_loss_scale
    _loss += _laudio_l2_loss * laudio_l2_loss_scale
    _loss += _audio_l2_loss * audio_l2_loss_scale
    _loss += _state_l2_loss * state_l2_loss_scale

    return _loss, _laudio_rec_loss, _audio_rec_loss, _laudio_disc_loss, _audio_disc_loss, _laudio_l2_loss, _audio_l2_loss, _state_l2_loss

def create_pose_animation(pose_start_frame, pose_frame_count, file_name):
    sequence_excerpt = pose_standardized[pose_start_frame:pose_start_frame + pose_frame_count]
    
    if standardize_poses:
        sequence_excerpt = pose_mean + sequence_excerpt * pose_std
    
    sequence_excerpt = np.reshape(sequence_excerpt, (-1, joint_count, joint_dim))
    sequence_excerpt_length = sequence_excerpt.shape[0]
    
    skel_sequence = skeleton.forward_kinematics(np.expand_dims(sequence_excerpt, axis=0), np.zeros((1, sequence_excerpt_length, 3)))
    skel_sequence = np.squeeze(skel_sequence)
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=1000 / pose_fps, loop=0)

def create_ref_audio(pose_start_frame, pose_frame_count, file_name):
    audio_start_sample = pose_start_frame * audio_samples_per_pose
    audio_sample_count = pose_frame_count * audio_samples_per_pose
    
    ref_audio = audio_standardized[audio_start_sample:audio_start_sample + audio_sample_count]
    if standardize_audio:
        ref_audio = audio_mean + ref_audio * audio_std
    audio_write(file_name, audio_save_sample_rate, ref_audio)

def create_pred_audio(pose_start_frame, pose_frame_count, pose_window_offset, file_name, audio_start_offset=0):
    
    if audio_start_offset > 0:
        audio_start_offset -= pose_start_frame
    
    pose_frame_count = (pose_frame_count // pose_window_length) * pose_window_length
    pose_window_offset = min(pose_window_offset, pose_window_length)

    audio_window_env = np.hanning(audio_window_length)
        
    pred_audio_sequence_length = pose_frame_count * audio_samples_per_pose
    pred_audio_sequence = np.zeros(shape=(pred_audio_sequence_length), dtype=np.float32)

    for pose_frame in range(pose_start_frame, pose_start_frame + pose_frame_count - pose_window_offset, pose_window_offset):
        pose_window = pose_standardized[pose_frame:pose_frame + pose_window_length]
        
        encoder_out = seq2seq_encoder(np.expand_dims(pose_window, axis=0), training=False)
        
        if pose_frame == pose_start_frame: # take sequence start audio window from reference audio
            audio_window_start_sample = (audio_start_offset + pose_frame) * audio_samples_per_pose - audio_window_offset
            audio_window_end_sample = audio_window_start_sample + audio_window_length
            audio_window = audio_standardized[audio_window_start_sample:audio_window_end_sample]
        else: # take sequence start audio window from predicted audio sequence
            audio_window_start_sample = (pose_frame - pose_start_frame) * audio_samples_per_pose - audio_window_offset
            audio_window_end_sample = audio_window_start_sample + audio_window_length
            audio_window = pred_audio_sequence[audio_window_start_sample:audio_window_end_sample] 
            
        audio_window_encoding = audio_encoder(np.expand_dims(audio_window, axis=0), training=False)
        decoder_in = tf.expand_dims(audio_window_encoding, axis=0)
        decoder_states_in = encoder_out

        for audio_window_index in range(audio_windows_per_pose_window):

            decoder_all_out = seq2seq_decoder([decoder_in] + decoder_states_in   , training=False)
            
            decoder_out = decoder_all_out[0]
            decoder_states_out = decoder_all_out[1:]
            
            laudio_window = decoder_out[0, -1, :]
            audio_window = audio_decoder(np.expand_dims(laudio_window, axis=0), training=False)
            audio_window = np.squeeze(audio_window, axis=0)

            audio_window = audio_window * audio_window_env

            pred_audio_sequence_start_sample = (pose_frame - pose_start_frame) * audio_samples_per_pose + audio_window_index * audio_window_offset
            pred_audio_sequence_end_sample = pred_audio_sequence_start_sample + audio_window_length
            pred_audio_sequence[pred_audio_sequence_start_sample:pred_audio_sequence_end_sample] += audio_window

            decoder_in = tf.reshape(laudio_window, shape=[1, 1, audio_latent_dimension])
            decoder_states_in = decoder_states_out
            
    if standardize_audio:
        pred_audio_sequence = audio_mean + pred_audio_sequence * audio_std

    # convert audio float array to audio int16 array with samples between -2**15 amd 2**15

    sample_mag = max( abs(np.min(pred_audio_sequence)), np.max(pred_audio_sequence) )

    if sample_mag > 1.0:
        pred_audio_sequence /= sample_mag
    
    pred_audio_int16_sequence = np.clip(pred_audio_sequence, -1.0, 1.0)
    pred_audio_int16_sequence = pred_audio_int16_sequence * audio_sample_scale
    pred_audio_int16_sequence = pred_audio_int16_sequence.astype(np.int16)

    audio_write(file_name, audio_save_sample_rate, pred_audio_int16_sequence)

def train(train_dataset, test_dataset, epochs):
    
    loss_history = {}
    loss_history["train loss"] = []
    loss_history["train laudio rec loss"] = []
    loss_history["train audio rec loss"] = []
    loss_history["train laudio disc loss"] = []
    loss_history["train audio disc loss"] = []
    loss_history["train laudio l2 loss"] = []
    loss_history["train audio l2 loss"] = []
    loss_history["train state l2 loss"] = []
    loss_history["test loss"] = []
    loss_history["test laudio rec loss"] = []
    loss_history["test audio rec loss"] = []
    loss_history["test laudio disc loss"] = []
    loss_history["test audio disc loss"] = []
    loss_history["test laudio l2 loss"] = []
    loss_history["test audio l2 loss"] = []
    loss_history["test state l2 loss"] = []

    for epoch in range(epochs):
        
        start = time.time()
        
        train_loss_per_epoch = []
        train_laudio_rec_loss_per_epoch = []
        train_audio_rec_loss_per_epoch = []
        train_laudio_disc_loss_per_epoch = []
        train_audio_disc_loss_per_epoch = []
        train_laudio_l2_loss_per_epoch = []
        train_audio_l2_loss_per_epoch = []
        train_state_l2_loss_per_epoch = []
        
        for (pose_batch, audio_batch, laudio_batch) in train_dataset:
            
            _ = laudio_disc_train_step(pose_batch, laudio_batch) 
            _loss, _laudio_rec_loss, _audio_rec_loss, _laudio_disc_loss, _audio_disc_loss, _laudio_l2_loss, _audio_l2_loss, _state_l2_loss = laudio_encoder_train_step(pose_batch, audio_batch, laudio_batch)

            _loss = np.array(_loss)
            _laudio_rec_loss = np.array(_laudio_rec_loss)
            _audio_rec_loss = np.array(_audio_rec_loss)
            _laudio_disc_loss = np.array(_laudio_disc_loss)
            _audio_disc_loss = np.array(_audio_disc_loss)
            _laudio_l2_loss = np.array(_laudio_l2_loss)
            _audio_l2_loss = np.array(_audio_l2_loss)
            _state_l2_loss = np.array(_state_l2_loss)
            
            train_loss_per_epoch.append(_loss)
            train_laudio_rec_loss_per_epoch.append(_laudio_rec_loss)
            train_audio_rec_loss_per_epoch.append(_audio_rec_loss)
            train_laudio_disc_loss_per_epoch.append(_laudio_disc_loss)
            train_audio_disc_loss_per_epoch.append(_audio_disc_loss)
            train_laudio_l2_loss_per_epoch.append(_laudio_l2_loss)
            train_audio_l2_loss_per_epoch.append(_audio_l2_loss)
            train_state_l2_loss_per_epoch.append(_state_l2_loss)
            
        train_loss_per_epoch = np.mean(np.array(train_loss_per_epoch))
        train_laudio_rec_loss_per_epoch = np.mean(np.array(train_laudio_rec_loss_per_epoch))
        train_audio_rec_loss_per_epoch = np.mean(np.array(train_audio_rec_loss_per_epoch))
        train_laudio_disc_loss_per_epoch = np.mean(np.array(train_laudio_disc_loss_per_epoch))
        train_audio_disc_loss_per_epoch = np.mean(np.array(train_audio_disc_loss_per_epoch))
        train_laudio_l2_loss_per_epoch = np.mean(np.array(train_laudio_l2_loss_per_epoch))
        train_audio_l2_loss_per_epoch = np.mean(np.array(train_audio_l2_loss_per_epoch))
        train_state_l2_loss_per_epoch = np.mean(np.array(train_state_l2_loss_per_epoch))
        
        test_loss_per_epoch = []
        test_laudio_rec_loss_per_epoch = []
        test_audio_rec_loss_per_epoch = []
        test_laudio_disc_loss_per_epoch = []
        test_audio_disc_loss_per_epoch = []
        test_laudio_l2_loss_per_epoch = []
        test_audio_l2_loss_per_epoch = []
        test_state_l2_loss_per_epoch = []
        
        for (pose_batch, audio_batch, laudio_batch) in test_dataset:
            
            _ = laudio_disc_train_step(pose_batch, laudio_batch) 
            _loss, _laudio_rec_loss, _audio_rec_loss, _laudio_disc_loss, _audio_disc_loss, _laudio_l2_loss, _audio_l2_loss, _state_l2_loss = laudio_encoder_test_step(pose_batch, audio_batch, laudio_batch)

            _loss = np.array(_loss)
            _laudio_rec_loss = np.array(_laudio_rec_loss)
            _audio_rec_loss = np.array(_audio_rec_loss)
            _laudio_disc_loss = np.array(_laudio_disc_loss)
            _audio_disc_loss = np.array(_audio_disc_loss)
            _laudio_l2_loss = np.array(_laudio_l2_loss)
            _audio_l2_loss = np.array(_audio_l2_loss)
            _state_l2_loss = np.array(_state_l2_loss)

            test_loss_per_epoch.append(_loss)
            test_laudio_rec_loss_per_epoch.append(_laudio_rec_loss)
            test_audio_rec_loss_per_epoch.append(_audio_rec_loss)
            test_laudio_disc_loss_per_epoch.append(_laudio_disc_loss)
            test_audio_disc_loss_per_epoch.append(_audio_disc_loss)
            test_laudio_l2_loss_per_epoch.append(_laudio_l2_loss)
            test_audio_l2_loss_per_epoch.append(_audio_l2_loss)
            test_state_l2_loss_per_epoch.append(_state_l2_loss)
            
        test_loss_per_epoch = np.mean(np.array(test_loss_per_epoch))
        test_laudio_rec_loss_per_epoch = np.mean(np.array(test_laudio_rec_loss_per_epoch))
        test_audio_rec_loss_per_epoch = np.mean(np.array(test_audio_rec_loss_per_epoch))
        test_laudio_disc_loss_per_epoch = np.mean(np.array(test_laudio_disc_loss_per_epoch))
        test_audio_disc_loss_per_epoch = np.mean(np.array(test_audio_disc_loss_per_epoch))
        test_laudio_l2_loss_per_epoch = np.mean(np.array(test_laudio_l2_loss_per_epoch))
        test_audio_l2_loss_per_epoch = np.mean(np.array(test_audio_l2_loss_per_epoch))
        test_state_l2_loss_per_epoch = np.mean(np.array(test_state_l2_loss_per_epoch))
        
        loss_history["train loss"].append(train_loss_per_epoch)
        loss_history["train laudio rec loss"].append(train_laudio_rec_loss_per_epoch)
        loss_history["train audio rec loss"].append(train_audio_rec_loss_per_epoch)
        loss_history["train laudio disc loss"].append(train_laudio_disc_loss_per_epoch)
        loss_history["train audio disc loss"].append(train_audio_disc_loss_per_epoch)
        loss_history["train laudio l2 loss"].append(train_laudio_l2_loss_per_epoch)
        loss_history["train audio l2 loss"].append(train_audio_l2_loss_per_epoch)
        loss_history["train state l2 loss"].append(train_state_l2_loss_per_epoch)
        loss_history["test loss"].append(test_loss_per_epoch)
        loss_history["test laudio rec loss"].append(test_laudio_rec_loss_per_epoch)
        loss_history["test audio rec loss"].append(test_audio_rec_loss_per_epoch)
        loss_history["test laudio disc loss"].append(test_laudio_disc_loss_per_epoch)
        loss_history["test audio disc loss"].append(test_audio_disc_loss_per_epoch)
        loss_history["test laudio l2 loss"].append(test_laudio_l2_loss_per_epoch)
        loss_history["test audio l2 loss"].append(test_audio_l2_loss_per_epoch)
        loss_history["test state l2 loss"].append(test_state_l2_loss_per_epoch)

        print ('epoch {} : train {:01.8f} test {:01.8f} recla {:01.8f} reca {:01.8f} ladisc {:01.8f} adisc {:01.8f} lal2 {:01.4f} al2 {:01.4f} stl2 {:01.4f} time {:01.2f}'.format(epoch + 1, train_loss_per_epoch, test_loss_per_epoch, train_laudio_rec_loss_per_epoch, train_audio_rec_loss_per_epoch, train_laudio_disc_loss_per_epoch, train_audio_disc_loss_per_epoch, train_laudio_l2_loss_per_epoch, train_audio_l2_loss_per_epoch, train_state_l2_loss_per_epoch, time.time()-start))
    
    return loss_history


loss_history = train(train_dataset, test_dataset, epochs)

utils.save_loss_as_csv(loss_history, "../results/seq2seq/history.csv")
utils.save_loss_as_image(loss_history, "../results/seq2seq/history.png")

seq2seq_encoder.save_weights("seq2seq/weights/seq2seq_encoder_epoch_{}".format(epochs))
seq2seq_decoder.save_weights("seq2seq/weights/seq2seq_decoder_epoch_{}".format(epochs))
laudio_disc.save_weights("seq2seq/weights/laudio_disc_epoch_{}".format(epochs))

create_pose_animation(4000, 1000, "../results/seq2seq/ref_pose_anim_4000.gif")
create_pose_animation(6000, 1000, "../results/seq2seq/ref_pose_anim_6000.gif")
create_pose_animation(14000, 1000, "../results/seq2seq/ref_pose_anim_14000.gif")
create_pose_animation(18000, 1000, "../results/seq2seq/ref_pose_anim_18000.gif")
create_pose_animation(22000, 1000, "../results/seq2seq/ref_pose_anim_22000.gif")

create_ref_audio(4000, 1000, "../results/seq2seq/ref_audio_4000.wav")
create_ref_audio(6000, 1000, "../results/seq2seq/ref_audio_6000.wav")
create_ref_audio(14000, 1000, "../results/seq2seq/ref_audio_14000.wav")
create_ref_audio(18000, 1000, "../results/seq2seq/ref_audio_18000.wav")
create_ref_audio(22000, 1000, "../results/seq2seq/ref_audio_22000.wav")

create_pred_audio(4000, 1000, 4, "../results/seq2seq/pred_audio_4000_epoch_{}.wav".format(epochs))
create_pred_audio(6000, 1000, 4, "../results/seq2seq/pred_audio_6000_epoch_{}.wav".format(epochs))
create_pred_audio(14000, 1000, 4, "../results/seq2seq/pred_audio_14000_epoch_{}.wav".format(epochs))
create_pred_audio(18000, 1000, 4, "../results/seq2seq/pred_audio_18000_epoch_{}.wav".format(epochs))
create_pred_audio(22000, 1000, 4, "../results/seq2seq/pred_audio_22000_epoch_{}.wav".format(epochs))



"""
Inference Tests with other mocap data
"""

mocap_data_path = "../data/movement_qualities_mocap.p"
mocap_valid_frame_ranges = [ [ 1000, 26400 ] ]              
mocap_valid_frame_count = np.sum([ range[1] - range[0] for range in mocap_valid_frame_ranges ])

mocap_data = MocapDataset(mocap_data_path, fps=pose_fps)
mocap_data.compute_positions()

skeleton = mocap_data.skeleton()
skeleton_joint_count = skeleton.num_joints()
skel_edge_list = utils.get_skeleton_edge_list(skeleton)
poseRenderer = PoseRenderer(skel_edge_list)

subject = np.random.choice(list(mocap_data.subjects()))
action = np.random.choice(list(mocap_data.subject_actions(subject)))
pose_sequence = mocap_data[subject][action]["rotations"]
total_sequence_length = pose_sequence.shape[0]
joint_count = pose_sequence.shape[1]
joint_dim = pose_sequence.shape[2]
pose_dim = joint_count * joint_dim
pose_sequence = np.reshape(pose_sequence, (-1, pose_dim))

if standardize_poses:
    pose_mean = np.mean(pose_sequence, axis=0)
    pose_std = np.std(pose_sequence, axis=0)
    pose_standardized = (pose_sequence - pose_mean) / (pose_std + 0.00001)
else:
    pose_standardized = pose_sequence   

for i in range(1000, 26000, 1000):
    print(i)
    create_pred_audio(i, 1000, 4, "Zac_003_inference_mocap_{}_audio_{}_epoch_{}.wav".format(i, i, epochs))
    create_pose_animation(i, 1000, "Zac_003_inference_pose_anim_{}.gif".format(i))
