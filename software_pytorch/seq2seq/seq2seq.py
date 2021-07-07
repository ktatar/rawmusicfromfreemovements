"""
seq2seq model translating mocap pose sequences into encoded audio window sequences
this model doesn't employ attention

Notes on discriminators:
the model uses two discriminators, one for distinguishing between real and fake audio encodings
and one for distinguishing between real and fake audio. The latter is identical to the one used
for the adversarial autoencoder. 
"""


# Imports

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
#device = 'cpu'

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
audio_file_path = "../data/sonification_audio"
#audio_file_path = "../data/improvisation_audio"
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
audio_encoder_model_file = "../results_pytorch/sonification/aae/models/ae_encoder" 
audio_decoder_model_file = "../results_pytorch/sonification/aae/models/ae_decoder" 
disc_audio_model_file = "../results_pytorch/sonification/aae/models/disc_audio" 
audio_encoder_weights_file = "../results_pytorch/sonification/aae/weights/ae_encoder_weights_epoch_200" 
audio_decoder_weights_file = "../results_pytorch/sonification/aae/weights/ae_decoder_weights_epoch_200" 
disc_audio_weights_file = "../results_pytorch/sonification/aae/weights/disc_audio_weights_epoch_200" 

# Model Configuration

# Seq2Seq Configuration
seq2seq_rnn_layer_sizes = [512, 512, 512] # this refers to the hidden size and must be the same for all layers!
final_decoder_activation = "linear"

# Latent Audio Discriminator
disc_laudio_activation_function = "ELU"
disc_laudio_dense_layer_sizes = [ 32, 32 ]
disc_laudio_dropout_prob = 0.0
disc_laudio_use_batch_normalization = True
disc_laudio_use_layer_normalization = False

# Save / Load Preprocesed Mocap and Audio Data
load_data = True
save_data = False
data_file_path = "../data/mocap_laudio_sonification_data"

# Save / Load Model Weights
save_models = False
save_tscript = False
load_weights = False
disc_laudio_model_file = "../results_pytorch/sonification/seq2seq/models/disc_laudio" 
seq2seq_encoder_model_file = "../results_pytorch/sonification/seq2seq/models/seq2seq_encoder" 
seq2seq_decoder_model_file = "../results_pytorch/sonification/seq2seq/models/seq2seq_decoder" 
disc_laudio_weights_file = "../results_pytorch/sonification/seq2seq/weights/disc_laudio_weights_epoch_200" 
seq2seq_encoder_weights_file = "../results_pytorch/sonification/seq2seq/weights/seq2seq_encoder_epoch_200"
seq2seq_decoder_weights_file = "../results_pytorch/sonification/seq2seq/weights/seq2seq_decoder_epoch_200"

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
disc_laudio_learning_rate = 4e-4

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
audio_encoder = torch.load("{}.pt".format(audio_encoder_model_file))
audio_decoder = torch.load("{}.pt".format(audio_decoder_model_file))
audio_disc = torch.load("{}.pt".format(disc_audio_model_file))

audio_encoder.load_state_dict(torch.load(audio_encoder_weights_file))
audio_decoder.load_state_dict(torch.load(audio_decoder_weights_file))
audio_disc.load_state_dict(torch.load(disc_audio_weights_file))

audio_encoder.eval()
audio_decoder.eval()
audio_disc.eval()

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
            audio_window = np.expand_dims(audio_window, axis=0)
            audio_window = torch.from_numpy(audio_window).to(device)
            
            audio_window_encoding = audio_encoder(audio_window)
    
            audio_window_encoding = np.squeeze(audio_window_encoding.detach().cpu().numpy(), axis=0)

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

# Create Dataset

sequence_count = pose_windows.shape[0]

class MovementAudioDataset(Dataset):
    def __init__(self, pose_windows, audio_window_sequences, audio_seq_encodings):
        self.pose_windows = pose_windows
        self.audio_window_sequences = audio_window_sequences
        self.audio_seq_encodings = audio_seq_encodings
    
    def __len__(self):
        return self.pose_windows.shape[0]
    
    def __getitem__(self, idx):
        return (self.pose_windows[idx, ...], self.audio_window_sequences[idx, ...], self.audio_seq_encodings[idx, ...])

    
full_dataset = MovementAudioDataset(pose_windows, audio_window_sequences, audio_seq_encodings)
dataset_size = len(full_dataset)

test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Create Models

# Latent Audio Discriminator

class DiscriminatorLAudio(nn.Module):
    def __init__(self):
        super(DiscriminatorLAudio, self).__init__()
        
        dense_layer_count = len(disc_laudio_dense_layer_sizes)
            
        layers = []
            
        in_size = audio_latent_dimension
            
        for layer_index in range(dense_layer_count):
            out_size = disc_laudio_dense_layer_sizes[layer_index]
                
            if disc_laudio_use_batch_normalization:
                layers.append(nn.BatchNorm1d(in_size))
                
            layers.append(nn.Linear(in_size, out_size))
                
            activation_function = getattr(nn, disc_laudio_activation_function)()  
            layers.append(activation_function)
                
            if disc_laudio_use_layer_normalization:
                layers.append(nn.LayerNorm())
            
            if disc_laudio_dropout_prob > 0.0:
                layers.append(nn.Dropout(p=disc_laudio_dropout_prob))
                    
            in_size = out_size
                
        layers.append(nn.Linear(in_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        yhat = self.model(x)
        return yhat                

disc_laudio = DiscriminatorLAudio().to(device)

print(disc_laudio)

"""
x = torch.rand((batch_size, audio_latent_dimension), dtype=torch.float32).to(device)
x.shape

x = disc_laudio(x)
x.shape
"""

if save_models == True:
    # save using pickle
    torch.save(disc_laudio, "{}.pth".format(disc_laudio_model_file))
    
    # save using onnx
    x = torch.zeros((batch_size, audio_latent_dimension)).to(device)
    torch.onnx.export(disc_laudio, x, "{}.onnx".format(disc_laudio_model_file))

if save_tscript == True:
    # save using TochScript
    x = torch.rand((batch_size, audio_latent_dimension), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(disc_laudio, x)
    script_module.save("{}.pt".format(disc_laudio_model_file))

if load_weights and disc_laudio_weights_file:
    disc_laudio.load_state_dict(torch.load(disc_laudio_weights_file))
    
# Seq2Seq Encoder

class Seq2SeqEncoder(nn.Module):
    def __init__(self):
        super(Seq2SeqEncoder, self).__init__()
        
        # create recurrent layers
        rnn_layer_size = seq2seq_rnn_layer_sizes[0]
        rnn_layer_count = len(seq2seq_rnn_layer_sizes)
        
        self.rnn = nn.LSTM(pose_dim, rnn_layer_size, rnn_layer_count, batch_first=True)
        
    def forward(self, x):
        
        #print("x 1 ", x.shape)
        
        x, (state_h,state_c) = self.rnn(x)
        
        #print("x 2 ", x.shape, " sh ", state_h.shape, " sc ", state_c.shape)
        
        return x, state_h, state_c

seq2seq_encoder = Seq2SeqEncoder().to(device)

print(seq2seq_encoder)

"""
x = torch.rand((batch_size, pose_window_length, pose_dim), dtype=torch.float32).to(device)
x.shape

x, state_h, state_c = seq2seq_encoder(x)
x.shape
state_h.shape
state_c.shape
"""

if save_models == True:
    # save using pickle
    torch.save(seq2seq_encoder, "{}.pth".format(seq2seq_encoder_model_file))
    
    # save using onnx
    x = torch.zeros((1, pose_window_length, pose_dim)).to(device)
    torch.onnx.export(seq2seq_encoder, x, "{}.onnx".format(seq2seq_encoder_model_file))

if save_tscript == True:
    # save using TochScript
    x = torch.rand((1, pose_window_length, pose_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(seq2seq_encoder, x)
    script_module.save("{}.pt".format(seq2seq_encoder_model_file))

if load_weights and seq2seq_encoder_weights_file:
    seq2seq_encoder.load_state_dict(torch.load(seq2seq_encoder_weights_file))

# Seq2Seq Encoder

class Seq2SeqDecoder(nn.Module):
    def __init__(self):
        super(Seq2SeqDecoder, self).__init__()
        
        # create recurrent layers
        
        rnn_layer_size = seq2seq_rnn_layer_sizes[0]
        rnn_layer_count = len(seq2seq_rnn_layer_sizes)
        
        self.rnn = nn.LSTM(audio_latent_dimension, rnn_layer_size, rnn_layer_count, batch_first=True)
        
        dense_layers = []
        dense_layers.append(nn.Linear(rnn_layer_size, audio_latent_dimension))
        
        if final_decoder_activation != "linear":
            activation_function = getattr(nn, final_decoder_activation)()  
            dense_layers.append(activation_function)
            
        self.dense = nn.Sequential(*dense_layers)
        
    def forward(self, x, state_h, state_c):

        #print("x 1 ", x.shape)

        x, (state_h,state_c) = self.rnn(x, (state_h.detach(), state_c.detach()))
        
        #print("x 2 ", x.shape, " sh ", state_h.shape, " sc ", state_c.shape)
        
        x = self.dense(x)
        
        #print("x 3 ", x.shape)
        
        return x, state_h, state_c

seq2seq_decoder = Seq2SeqDecoder().to(device)

print(seq2seq_decoder)

"""
x = torch.rand((batch_size, pose_window_length, pose_dim), dtype=torch.float32).to(device)
x.shape

x, state_h, state_c = seq2seq_encoder(x)
x.shape
state_h.shape
state_c.shape

x = torch.rand((batch_size, audio_windows_per_pose_window, audio_latent_dimension), dtype=torch.float32).to(device)
x.shape

x, state_h, state_c = seq2seq_decoder(x, state_h, state_c)
x.shape
state_h.shape
state_c.shape
"""

if save_models == True:
    # save using pickle
    torch.save(seq2seq_decoder, "{}.pth".format(seq2seq_decoder_model_file))
    
    x = torch.zeros((1, audio_windows_per_pose_window, audio_latent_dimension)).to(device)
    state_h = torch.zeros((len(seq2seq_rnn_layer_sizes), 1, seq2seq_rnn_layer_sizes[0])).to(device)
    state_c = torch.zeros((len(seq2seq_rnn_layer_sizes), 1, seq2seq_rnn_layer_sizes[0])).to(device)
    torch.onnx.export(seq2seq_decoder, (x, state_h, state_c), "{}.onnx".format(seq2seq_decoder_model_file))

if save_tscript == True:
    # save using TochScript
    x = torch.zeros((1, audio_windows_per_pose_window, audio_latent_dimension)).to(device)
    state_h = torch.zeros((len(seq2seq_rnn_layer_sizes), 1, seq2seq_rnn_layer_sizes[0])).to(device)
    state_c = torch.zeros((len(seq2seq_rnn_layer_sizes), 1, seq2seq_rnn_layer_sizes[0])).to(device)
    script_module = torch.jit.trace(seq2seq_decoder, (x, state_h, state_c))
    script_module.save("{}.pt".format(seq2seq_decoder_model_file))

if load_weights and seq2seq_encoder_weights_file:
    seq2seq_decoder.load_state_dict(torch.load(seq2seq_encoder_weights_file))

# Training

# loss functions

cross_entropy = nn.BCELoss()

def disc_loss(d_x, g_z, smoothing_factor = 0.9):
    
    ones = torch.ones_like(d_x).to(device)
    zeros = torch.zeros_like(g_z).to(device)
    
    real_loss = cross_entropy(d_x, ones * smoothing_factor) # If we feed the discriminator with real images, we assume they all are the right pictures --> Because of that label == 1
    fake_loss = cross_entropy(g_z, zeros) # Each noise we feed in are fakes image --> Because of that labels are 0. 
    total_loss = real_loss + fake_loss
    
    return total_loss

def disc_gen_loss(dx_of_gx):
    ones = torch.ones_like(dx_of_gx).to(device)
    return cross_entropy(dx_of_gx, ones) 

def l2_loss(z):
    flat_z = torch.flatten(z)
    _loss = torch.mean(flat_z**2)
    
    return _loss

# autoencoder audio reconstruction loss
def rec_loss(y, yhat):
    flat_y = torch.flatten(y)
    flat_yhat = torch.flatten(yhat)
    _loss = torch.mean((flat_y-flat_yhat)**2)
    
    return _loss

# optimizers

disc_laudio_optimizer = torch.optim.Adam(disc_laudio.parameters(), lr=disc_laudio_learning_rate)
seq2seq_optimizer = torch.optim.Adam(list(seq2seq_encoder.parameters()) + list(seq2seq_decoder.parameters()), lr=seq2seq_learning_rate)

# Training and Testing Functions

def laudio_disc_train_step(pose_batch, laudio_batch):

    #print("laudio_disc_train_step")    
    
    #print("pose_batch s ", pose_batch.shape)
    #print("laudio_batch s ", laudio_batch.shape)

    real_laudio = laudio_batch[:, 1:, :]
    
    #print("real_laudio s ", real_laudio.shape)
    
    with torch.no_grad():
        enc_out, enc_state_h, env_state_c = seq2seq_encoder(pose_batch)
        dec_out, _, _ = seq2seq_decoder( laudio_batch[:, :-1, :], enc_state_h, env_state_c)

    fake_laudio = dec_out
    
    #print("real_laudio s ", real_laudio.shape)
    #print("fake_laudio s ", fake_laudio.shape)
    
    laudio_disc_real_output =  disc_laudio(torch.reshape(real_laudio, (-1, audio_latent_dimension)))
    laudio_disc_fake_output =  disc_laudio(torch.reshape(fake_laudio, (-1, audio_latent_dimension)))
    
    #print("laudio_disc_real_output s ", laudio_disc_real_output.shape)
    #print("laudio_disc_fake_output s ", laudio_disc_fake_output.shape)
    
    #torch.reshape(laudio_disc_real_output, (-1, audio_windows_per_pose_window, 1))
    #torch.reshape(laudio_disc_fake_output, (-1, audio_windows_per_pose_window, 1))
    
    #laudio_disc_real_output.view(-1, audio_windows_per_pose_window, 1)
    #laudio_disc_fake_output.view(-1, audio_windows_per_pose_window, 1)
    
    #print("laudio_disc_real_output2 s ", laudio_disc_real_output.shape)
    #print("laudio_disc_fake_output2 s ", laudio_disc_fake_output.shape)

    _loss = disc_loss(laudio_disc_real_output, laudio_disc_fake_output)
    
    disc_laudio_optimizer.zero_grad()
    _loss.backward()
    disc_laudio_optimizer.step()
    
    return _loss


def laudio_disc_test_step(pose_batch, laudio_batch):
    
    #print("laudio_disc_test_step")    
    
    #print("pose_batch s ", pose_batch.shape)
    #print("laudio_batch s ", laudio_batch.shape)

    real_laudio = laudio_batch[:, 1:, :]
    
    #print("real_laudio s ", real_laudio.shape)
    
    with torch.no_grad():
        enc_out, enc_state_h, env_state_c = seq2seq_encoder(pose_batch)
        dec_out, _, _ = seq2seq_decoder( laudio_batch[:, :-1, :], enc_state_h, env_state_c)

        fake_laudio = dec_out
    
        #print("real_laudio s ", real_laudio.shape)
        #print("fake_laudio s ", fake_laudio.shape)
    
        real_laudio = torch.reshape(real_laudio, (-1, audio_latent_dimension))
        fake_laudio = torch.reshape(fake_laudio, (-1, audio_latent_dimension))
    
        #print("real_laudio2 s ", real_laudio.shape)
        #print("fake_laudio2 s ", fake_laudio.shape)
    
        laudio_disc_real_output =  disc_laudio(real_laudio)
        laudio_disc_fake_output =  disc_laudio(fake_laudio)
    
        #print("laudio_disc_real_output s ", laudio_disc_real_output.shape)
        #print("laudio_disc_fake_output s ", laudio_disc_fake_output.shape)
    
        torch.reshape(laudio_disc_real_output, (-1, audio_windows_per_pose_window, 1))
        torch.reshape(laudio_disc_fake_output, (-1, audio_windows_per_pose_window, 1))
    
        #print("laudio_disc_real_output2 s ", laudio_disc_real_output.shape)
        #print("laudio_disc_fake_output2 s ", laudio_disc_fake_output.shape)

    _loss = disc_loss(laudio_disc_real_output, laudio_disc_fake_output)
    
    return _loss

"""
def laudio_encoder_train_step(pose_batch, audio_batch, laudio_batch):
    
    #print("laudio_encoder_train_step")
    
    real_laudio = laudio_batch
    real_audio = audio_batch
    
    #print("real_laudio s ", real_laudio.shape)
    #print("real_audio s ", real_audio.shape)
        
    enc_out, enc_state_h, enc_state_c = seq2seq_encoder(pose_batch)
    
    #print("enc_out s ", enc_out.shape)
    #print("enc_state_h s ", enc_state_h.shape)
    #print("enc_state_c s ", enc_state_c.shape)
    
    dec_out, dec_state_h, dec_state_c = seq2seq_decoder(laudio_batch[:,:-1,:], enc_state_h, enc_state_c)
    
    #print("dec_out s ", dec_out.shape)
    #print("dec_state_h s ", dec_state_h.shape)
    #print("dec_state_c s ", dec_state_c.shape)   
    
    pred_laudio = dec_out
    
    #print("pred_laudio s ", pred_laudio.shape) 
    
    pred_laudio = pred_laudio.view(-1, audio_latent_dimension )
    
    #print("pred_laudio2 s ", pred_laudio.shape) 
    
    # laudio disc inference
    laudio_disc_fake_output = disc_laudio(pred_laudio)
    
    #print("laudio_disc_fake_output s ", laudio_disc_fake_output.shape) 
    
    pred_audio = audio_decoder(pred_laudio)
    
    #print("pred_audio s ", pred_audio.shape) 

    # audio disc inference
    audio_disc_fake_output = audio_disc(pred_audio)
        
    #print("audio_disc_fake_output s ", audio_disc_fake_output.shape) 
    
    pred_laudio = pred_laudio.view(-1, audio_windows_per_pose_window, audio_latent_dimension )
    
    #print("pred_laudio3 s ", pred_laudio.shape) 
    
    pred_audio = pred_audio.view(-1, audio_windows_per_pose_window, audio_window_length )
    
    #print("pred_audio2 s ", pred_audio.shape) 

    _laudio_rec_loss = rec_loss(laudio_batch[:, 1:, :], pred_laudio)
    _audio_rec_loss = rec_loss(audio_batch[:, 1:, :], pred_audio)
    _laudio_disc_loss = disc_gen_loss(laudio_disc_fake_output)
    _audio_disc_loss = disc_gen_loss(audio_disc_fake_output)
    _laudio_l2_loss = l2_loss(pred_laudio)
    _audio_l2_loss = l2_loss(pred_audio)
    _state_l2_loss = l2_loss(torch.cat((enc_state_h, enc_state_c), 0))
                
    _loss = 0.0
    #_loss += _laudio_rec_loss * laudio_rec_loss_scale
    #_loss += _audio_rec_loss * audio_rec_loss_scale
    #_loss += _laudio_disc_loss * laudio_disc_loss_scale
    #_loss += _audio_disc_loss * audio_disc_loss_scale
    #_loss += _laudio_l2_loss * laudio_l2_loss_scale
    #_loss += _audio_l2_loss * audio_l2_loss_scale
    #_loss += _state_l2_loss * state_l2_loss_scale

    #_loss += _laudio_rec_loss
    #_loss += _audio_rec_loss
    _loss += _laudio_disc_loss

    seq2seq_optimizer.zero_grad()
    _loss.backward()
    seq2seq_optimizer.step()
    
    return _loss, _laudio_rec_loss, _audio_rec_loss, _laudio_disc_loss, _audio_disc_loss, _laudio_l2_loss, _audio_l2_loss, _state_l2_loss
"""

def laudio_encoder_train_step(pose_batch, audio_batch, laudio_batch):
    
    #print("laudio_encoder_train_step")
    
    real_laudio = laudio_batch
    real_audio = audio_batch
    
    #print("real_laudio s ", real_laudio.shape)
    #print("real_audio s ", real_audio.shape)
        
    enc_out, enc_state_h, enc_state_c = seq2seq_encoder(pose_batch)
    
    #print("enc_out s ", enc_out.shape)
    #print("enc_state_h s ", enc_state_h.shape)
    #print("enc_state_c s ", enc_state_c.shape)
    
    dec_out, dec_state_h, dec_state_c = seq2seq_decoder(laudio_batch[:,:-1,:], enc_state_h, enc_state_c)
    
    #print("dec_out s ", dec_out.shape)
    #print("dec_state_h s ", dec_state_h.shape)
    #print("dec_state_c s ", dec_state_c.shape)   
    
    pred_laudio = dec_out
    
    #print("pred_laudio s ", pred_laudio.shape) 
    #print("real laudio s ", laudio_batch[:, 1:, :].shape)
    
    #pred_laudio = pred_laudio.view(-1, audio_latent_dimension )
    
    #print("pred_laudio2 s ", pred_laudio.shape) 
    
    # laudio disc inference
    # laudio_disc_fake_output = disc_laudio(pred_laudio)
    
    laudio_disc_fake_output = disc_laudio(torch.reshape(pred_laudio,(-1, audio_latent_dimension)))
    
    #print("laudio_disc_fake_output s ", laudio_disc_fake_output.shape) 
    
    pred_audio = audio_decoder(torch.reshape(pred_laudio, (-1, audio_latent_dimension)))
    
    #print("pred_audio s ", pred_audio.shape) 
    #print("real_audio s ", audio_batch[:, 1:, :].shape) 

    # audio disc inference
    audio_disc_fake_output = audio_disc(pred_audio)
        
    #print("audio_disc_fake_output s ", audio_disc_fake_output.shape) 
    
    #pred_laudio = pred_laudio.view(-1, audio_windows_per_pose_window, audio_latent_dimension )
    
    #print("pred_laudio3 s ", pred_laudio.shape) 
    
    #pred_audio = pred_audio.view(-1, audio_windows_per_pose_window, audio_window_length )
    
    #print("pred_audio2 s ", pred_audio.shape) 

    _laudio_rec_loss = rec_loss(laudio_batch[:, 1:, :], pred_laudio)
    _audio_rec_loss = rec_loss(audio_batch[:, 1:, :], torch.reshape(pred_audio,(-1, audio_windows_per_pose_window, audio_window_length)))
    _laudio_disc_loss = disc_gen_loss(laudio_disc_fake_output)
    _audio_disc_loss = disc_gen_loss(audio_disc_fake_output)
    _laudio_l2_loss = l2_loss(pred_laudio)
    _audio_l2_loss = l2_loss(pred_audio)
    _state_l2_loss = l2_loss(torch.cat((enc_state_h, enc_state_c), 0))

                
    _loss = 0.0
    _loss += _laudio_rec_loss * laudio_rec_loss_scale
    _loss += _audio_rec_loss * audio_rec_loss_scale
    _loss += _laudio_disc_loss * laudio_disc_loss_scale
    _loss += _audio_disc_loss * audio_disc_loss_scale
    _loss += _laudio_l2_loss * laudio_l2_loss_scale
    _loss += _audio_l2_loss * audio_l2_loss_scale
    _loss += _state_l2_loss * state_l2_loss_scale


    seq2seq_optimizer.zero_grad()
    _loss.backward()
    seq2seq_optimizer.step()
    
    return _loss, _laudio_rec_loss, _audio_rec_loss, _laudio_disc_loss, _audio_disc_loss, _laudio_l2_loss, _audio_l2_loss, _state_l2_loss


def laudio_encoder_test_step(pose_batch, audio_batch, laudio_batch):
    #print("laudio_encoder_test_step")
    
    real_laudio = laudio_batch
    real_audio = audio_batch
    
    #print("real_laudio s ", real_laudio.shape)
    #print("real_audio s ", real_audio.shape)
        
    with torch.no_grad():
        enc_out, enc_state_h, enc_state_c = seq2seq_encoder(pose_batch)
    
        #print("enc_out s ", enc_out.shape)
        #print("enc_state_h s ", enc_state_h.shape)
        #print("enc_state_c s ", enc_state_c.shape)
    
        dec_out, dec_state_h, dec_state_c = seq2seq_decoder(laudio_batch[:,:-1,:], enc_state_h, enc_state_c)
    
        #print("dec_out s ", dec_out.shape)
        #print("dec_state_h s ", dec_state_h.shape)
        #print("dec_state_c s ", dec_state_c.shape)   
    
        pred_laudio = dec_out
    
        #print("pred_laudio s ", pred_laudio.shape) 
    
        pred_laudio = pred_laudio.view(-1, audio_latent_dimension )
    
        #print("pred_laudio2 s ", pred_laudio.shape) 
    
        # laudio disc inference
        laudio_disc_fake_output = disc_laudio(pred_laudio)
    
        #print("laudio_disc_fake_output s ", laudio_disc_fake_output.shape) 
    
        pred_audio = audio_decoder(pred_laudio)
    
        #print("pred_audio s ", pred_audio.shape) 

        # audio disc inference
        audio_disc_fake_output = audio_disc(pred_audio)
        
        #print("audio_disc_fake_output s ", audio_disc_fake_output.shape) 
    
        """
        pred_laudio = pred_laudio.view(-1, audio_windows_per_pose_window, audio_latent_dimension )
    
        print("pred_laudio3 s ", pred_laudio.shape) 
    
        pred_audio = pred_audio.view(-1, audio_windows_per_pose_window, audio_window_length )
    
        print("pred_audio2 s ", pred_audio.shape) 
        """
        
        _laudio_rec_loss = rec_loss(laudio_batch[:, 1:, :], pred_laudio)
        _audio_rec_loss = rec_loss(audio_batch[:, 1:, :], pred_audio)
        _laudio_disc_loss = disc_gen_loss(laudio_disc_fake_output)
        _audio_disc_loss = disc_gen_loss(audio_disc_fake_output)
        _laudio_l2_loss = l2_loss(pred_laudio)
        _audio_l2_loss = l2_loss(pred_audio)
        _state_l2_loss = l2_loss(torch.cat((enc_state_h, enc_state_c), 0))
                
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
    
    seq2seq_encoder.eval()
    seq2seq_decoder.eval()
    
    if audio_start_offset > 0:
        audio_start_offset -= pose_start_frame
    
    pose_frame_count = (pose_frame_count // pose_window_length) * pose_window_length
    pose_window_offset = min(pose_window_offset, pose_window_length)

    audio_window_env = np.hanning(audio_window_length)
        
    pred_audio_sequence_length = pose_frame_count * audio_samples_per_pose
    pred_audio_sequence = np.zeros(shape=(pred_audio_sequence_length), dtype=np.float32)

    for pose_frame in range(pose_start_frame, pose_start_frame + pose_frame_count - pose_window_offset, pose_window_offset):
        pose_window = pose_standardized[pose_frame:pose_frame + pose_window_length]
        pose_window = np.expand_dims(pose_window, axis=0)
        pose_window = torch.from_numpy(pose_window).to(device)
        
        with torch.no_grad():
            enc_out, enc_state_h_out, enc_state_c_out = seq2seq_encoder(pose_window)
        
        if pose_frame == pose_start_frame: # take sequence start audio window from reference audio
            audio_window_start_sample = (audio_start_offset + pose_frame) * audio_samples_per_pose - audio_window_offset
            audio_window_end_sample = audio_window_start_sample + audio_window_length
            audio_window = audio_standardized[audio_window_start_sample:audio_window_end_sample]
        else: # take sequence start audio window from predicted audio sequence
            audio_window_start_sample = (pose_frame - pose_start_frame) * audio_samples_per_pose - audio_window_offset
            audio_window_end_sample = audio_window_start_sample + audio_window_length
            audio_window = pred_audio_sequence[audio_window_start_sample:audio_window_end_sample] 
            
        audio_window = np.expand_dims(audio_window, axis=0)
        audio_window = torch.from_numpy(audio_window).to(device)
        
        with torch.no_grad():
            audio_window_encoding = audio_encoder(audio_window)

        audio_window_encoding = torch.unsqueeze(audio_window_encoding, 0)
        dec_in = audio_window_encoding
        dec_state_h_in = enc_state_h_out
        dec_state_c_in = enc_state_c_out

        for audio_window_index in range(audio_windows_per_pose_window):
            
            with torch.no_grad():
                dec_out, dec_state_h_out, dec_state_c_out = seq2seq_decoder(dec_in, dec_state_h_in, dec_state_c_in)

            laudio_window = dec_out[0, -1, :]
            laudio_window = torch.unsqueeze(laudio_window, 0)
            
            with torch.no_grad():
                audio_window = audio_decoder(laudio_window)
                
            audio_window = np.squeeze(audio_window.detach().cpu().numpy())
            audio_window = audio_window * audio_window_env

            pred_audio_sequence_start_sample = (pose_frame - pose_start_frame) * audio_samples_per_pose + audio_window_index * audio_window_offset
            pred_audio_sequence_end_sample = pred_audio_sequence_start_sample + audio_window_length
            pred_audio_sequence[pred_audio_sequence_start_sample:pred_audio_sequence_end_sample] += audio_window
            
            dec_in = laudio_window.view(1, 1, audio_latent_dimension)
            dec_state_h_in = dec_state_h_out
            dec_state_c_in = dec_state_c_out
            
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
    
    seq2seq_encoder.train()
    seq2seq_decoder.train()
    
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
    
    loss_history["debug loss"] = []

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
        
        debug_loss_per_epoch = []
        
        for (pose_batch, audio_batch, laudio_batch) in train_dataset:
            
            pose_batch = pose_batch.to(device)
            audio_batch = audio_batch.to(device)
            laudio_batch = laudio_batch.to(device)
            
            #print("train ep ", epoch, " ps ", pose_batch.shape, " as ", audio_batch.shape, " la ", laudio_batch.shape)

            _debug_loss = laudio_disc_train_step(pose_batch, laudio_batch) 
            _loss, _laudio_rec_loss, _audio_rec_loss, _laudio_disc_loss, _audio_disc_loss, _laudio_l2_loss, _audio_l2_loss, _state_l2_loss = laudio_encoder_train_step(pose_batch, audio_batch, laudio_batch)

            _loss = _loss.detach().cpu().numpy()
            _laudio_rec_loss = _laudio_rec_loss.detach().cpu().numpy()
            _audio_rec_loss = _audio_rec_loss.detach().cpu().numpy()
            _laudio_disc_loss = _laudio_disc_loss.detach().cpu().numpy()
            _audio_disc_loss = _audio_disc_loss.detach().cpu().numpy()
            _laudio_l2_loss = _laudio_l2_loss.detach().cpu().numpy()
            _audio_l2_loss = _audio_l2_loss.detach().cpu().numpy()
            _state_l2_loss = _state_l2_loss.detach().cpu().numpy()
            
            _debug_loss = _debug_loss.detach().cpu().numpy()
            
            train_loss_per_epoch.append(_loss)
            train_laudio_rec_loss_per_epoch.append(_laudio_rec_loss)
            train_audio_rec_loss_per_epoch.append(_audio_rec_loss)
            train_laudio_disc_loss_per_epoch.append(_laudio_disc_loss)
            train_audio_disc_loss_per_epoch.append(_audio_disc_loss)
            train_laudio_l2_loss_per_epoch.append(_laudio_l2_loss)
            train_audio_l2_loss_per_epoch.append(_audio_l2_loss)
            train_state_l2_loss_per_epoch.append(_state_l2_loss)
            
            debug_loss_per_epoch.append(_debug_loss)
            
            
            
        train_loss_per_epoch = np.mean(np.array(train_loss_per_epoch))
        train_laudio_rec_loss_per_epoch = np.mean(np.array(train_laudio_rec_loss_per_epoch))
        train_audio_rec_loss_per_epoch = np.mean(np.array(train_audio_rec_loss_per_epoch))
        train_laudio_disc_loss_per_epoch = np.mean(np.array(train_laudio_disc_loss_per_epoch))
        train_audio_disc_loss_per_epoch = np.mean(np.array(train_audio_disc_loss_per_epoch))
        train_laudio_l2_loss_per_epoch = np.mean(np.array(train_laudio_l2_loss_per_epoch))
        train_audio_l2_loss_per_epoch = np.mean(np.array(train_audio_l2_loss_per_epoch))
        train_state_l2_loss_per_epoch = np.mean(np.array(train_state_l2_loss_per_epoch))
        
        debug_loss_per_epoch = np.mean(np.array(debug_loss_per_epoch))
        
        print("debug_loss_per_epoch ", debug_loss_per_epoch)
        
        """
        test_loss_per_epoch = []
        test_laudio_rec_loss_per_epoch = []
        test_audio_rec_loss_per_epoch = []
        test_laudio_disc_loss_per_epoch = []
        test_audio_disc_loss_per_epoch = []
        test_laudio_l2_loss_per_epoch = []
        test_audio_l2_loss_per_epoch = []
        test_state_l2_loss_per_epoch = []
        
        for (pose_batch, audio_batch, laudio_batch) in test_dataset:
            
            pose_batch = pose_batch.to(device)
            audio_batch = audio_batch.to(device)
            laudio_batch = laudio_batch.to(device)
            
            _ = laudio_disc_train_step(pose_batch, laudio_batch) 
            _loss, _laudio_rec_loss, _audio_rec_loss, _laudio_disc_loss, _audio_disc_loss, _laudio_l2_loss, _audio_l2_loss, _state_l2_loss = laudio_encoder_test_step(pose_batch, audio_batch, laudio_batch)

            _loss = _loss.detach().cpu().numpy()
            _laudio_rec_loss = _laudio_rec_loss.detach().cpu().numpy()
            _audio_rec_loss = _audio_rec_loss.detach().cpu().numpy()
            _laudio_disc_loss = _laudio_disc_loss.detach().cpu().numpy()
            _audio_disc_loss = _audio_disc_loss.detach().cpu().numpy()
            _laudio_l2_loss = _laudio_l2_loss.detach().cpu().numpy()
            _audio_l2_loss = _audio_l2_loss.detach().cpu().numpy()
            _state_l2_loss = _state_l2_loss.detach().cpu().numpy()

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
        """
        
        test_loss_per_epoch = 0.0
        test_laudio_rec_loss_per_epoch = 0.0
        test_audio_rec_loss_per_epoch = 0.0
        test_laudio_disc_loss_per_epoch = 0.0
        test_audio_disc_loss_per_epoch = 0.0
        test_laudio_l2_loss_per_epoch = 0.0
        test_audio_l2_loss_per_epoch = 0.0
        test_state_l2_loss_per_epoch = 0.0
        
        print ('epoch {} : train {:01.8f} test {:01.8f} recla {:01.8f} reca {:01.8f} ladisc {:01.8f} adisc {:01.8f} lal2 {:01.4f} al2 {:01.4f} stl2 {:01.4f} time {:01.2f}'.format(epoch + 1, train_loss_per_epoch, test_loss_per_epoch, train_laudio_rec_loss_per_epoch, train_audio_rec_loss_per_epoch, train_laudio_disc_loss_per_epoch, train_audio_disc_loss_per_epoch, train_laudio_l2_loss_per_epoch, train_audio_l2_loss_per_epoch, train_state_l2_loss_per_epoch, time.time()-start))
    
    return loss_history

loss_history = train(train_dataloader, test_dataloader, epochs)


utils.save_loss_as_csv(loss_history, "../results/seq2seq/history.csv")
utils.save_loss_as_image(loss_history, "../results/seq2seq/history.png")

torch.save(disc_laudio.state_dict(), "../results_pytorch/sonification/seq2seq/weights/disc_laudio_weights_epoch_{}".format(epochs))
torch.save(seq2seq_encoder.state_dict(), "../results_pytorch/sonification/seq2seq/weights/seq2seq_encoder_weights_epoch_{}".format(epochs))
torch.save(seq2seq_decoder.state_dict(), "../results_pytorch/sonification/seq2seq/weights/seq2seq_decoder_weights_epoch_{}".format(epochs))


create_pose_animation(4000, 1000, "../results/seq2seq/ref_pose_anim_4000.gif")
create_pose_animation(6000, 1000, "../results/seq2seq/ref_pose_anim_6000.gif")
create_pose_animation(14000, 1000, "../results/seq2seq/ref_pose_anim_14000.gif")
create_pose_animation(18000, 1000, "../results/seq2seq/ref_pose_anim_18000.gif")
create_pose_animation(22000, 1000, "../results/seq2seq/ref_pose_anim_22000.gif")

create_ref_audio(4000, 1000, "../results_pytorch/sonification/seq2seq/audio/ref_audio_4000.wav")
create_ref_audio(6000, 1000, "../results/seq2seq/ref_audio_6000.wav")
create_ref_audio(14000, 1000, "../results/seq2seq/ref_audio_14000.wav")
create_ref_audio(18000, 1000, "../results/seq2seq/ref_audio_18000.wav")
create_ref_audio(22000, 1000, "../results/seq2seq/ref_audio_22000.wav")

create_pred_audio(4000, 1000, 4, "../results_pytorch/sonification/seq2seq/audio/pred_audio_4000_epoch_{}.wav".format(epochs))
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
