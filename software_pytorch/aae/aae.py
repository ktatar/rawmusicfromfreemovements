"""
an adversarial autoencoder for raw audio

Notes on discriminators:
the model uses two discriminators, one for distinguishing between real and fake prior distributions
and one for distinguishing between real and fake audio.

Notes on dilated convolutions:
the model combines dilated convolution with regular convolution.
It does so my creating branched paths for each convolution step where dilation is greater than 0,
the branched paths then join again by adding the results of the regular convolution and dilated convolution.
"""

# Imports

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import OrderedDict
import simpleaudio as sa
from scipy.io.wavfile import read as audio_read
from scipy.io.wavfile import write as audio_write
import numpy as np
import glob
from matplotlib import pyplot as plt
import os, time
import json
from common import utils


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
#device = 'cpu'

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
audio_window_offset = 32
batch_size = 128
epochs = 200
train_percentage = 0.9
test_percentage = 0.1
ae_learning_rate = 1e-4
disc_audio_learning_rate = 4e-4
disc_prior_learning_rate = 4e-4
ae_rec_loss_scale = 1.0
ae_disc_audio_loss_scale = 1.0
ae_disc_prior_loss_scale = 0.1
ae_l1_loss_scale = 0.0
ae_l2_loss_scale = 0.01
disc_prior_label_smoothing = 0.9
disc_audio_label_smoothing = 0.9
model_save_interval = 50
save_history = False

# Model Configuration

# Autoencoder Configuration
latent_dim = 32
ae_activation_function = "LeakyReLU"
ae_conv_filter_counts = [8, 16, 32, 64]
ae_conv_kernel_sizes = [7, 7, 7, 7] # important: use odd kernel sizes otherwise padding calculation fails
ae_conv_strides = [4, 4, 4, 4]
ae_conv_dilations = [0, 0, 0, 0]
ae_dense_layer_sizes = [32]
ae_dropout_prob = 0.0
ae_use_batch_normalization = True
ae_use_layer_normalization = False

# Prior Discriminator Configuration
disc_prior_activation_function = "LeakyReLU"
disc_prior_dense_layer_sizes = [32, 32]
disc_prior_dropout_prob = 0.0
disc_prior_use_batch_normalization = True
disc_prior_use_layer_normalization = False

# Audio Discriminator Configuration
disc_audio_activation_function = "LeakyReLU"
disc_audio_conv_filter_counts = [8, 16, 32, 64]
disc_audio_conv_kernel_sizes = [7, 7, 7, 7] # important: use odd kernel sizes otherwise padding calculation fails
disc_audio_conv_strides = [4, 4, 4, 4]
disc_audio_conv_dilations = [0, 0, 0, 0]
disc_audio_dense_layer_sizes = [32]
disc_audio_dropout_prob = 0.0
disc_audio_use_batch_normalization = True
disc_audio_use_layer_normalization = False

# Save / Load Model Weights
save_models = True
save_tscript = True
save_weights = False
load_weights = True
disc_prior_model_file = "../results_pytorch/sonification/aae/models/disc_prior"
disc_audio_model_file = "../results_pytorch/sonification/aae/models/disc_audio"
ae_encoder_model_file = "../results_pytorch/sonification/aae/models/ae_encoder"
ae_decoder_model_file = "../results_pytorch/sonification/aae/models/ae_decoder"
disc_prior_weights_file = "../results_pytorch/sonification/aae/weights/disc_prior_weights_epoch_200"
disc_audio_weights_file = "../results_pytorch/sonification/aae/weights/disc_audio_weights_epoch_200"
ae_encoder_weights_file = "../results_pytorch/sonification/aae/weights/ae_encoder_weights_epoch_200"
ae_decoder_weights_file = "../results_pytorch/sonification/aae/weights/ae_decoder_weights_epoch_200"

# Save Audio Examples
save_audio = False
audio_save_interval = 100
audio_save_start_times = [ 40.0, 120.0, 300.0, 480.0  ] # in seconds
audio_save_duration = 10.0
audio_traverse_start_window = 0
audio_traverse_end_window = 1000
audio_traverse_window_count = 10
audio_traverse_interpolation_count = 100
audio_result_path = "../results_pytorch/sonification/aae/audio"


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

class AudioDataset(Dataset):
    def __init__(self, audio_windows):
        self.audio_windows = audio_windows
    
    def __len__(self):
        return self.audio_windows.shape[0]
    
    def __getitem__(self, idx):
        return self.audio_windows[idx, ...]

full_dataset = AudioDataset(audio_training_data)
dataset_size = len(full_dataset)

test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


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
        file_name = "{}/ref_audio_{}.wav".format(audio_result_path, audio_start_time)
        create_ref_sonifications(audio_start_time, audio_save_duration, file_name)

# Create Models

# function returning normal distributed random data 
# serves as reference for the discriminator to distinguish the encoders prior from

def sample_normal(shape):
    return torch.tensor(np.random.normal(size=shape), dtype=torch.float32).to(device)

class DiscriminatorPrior(nn.Module):
    def __init__(self):
        super(DiscriminatorPrior, self).__init__()
        
        dense_layer_count = len(disc_prior_dense_layer_sizes)
        
        layers = []
        
        in_size = latent_dim

        for layer_index in range(dense_layer_count):
            
            out_size = disc_prior_dense_layer_sizes[layer_index]
            
            if disc_prior_use_batch_normalization:
                    layers.append(nn.BatchNorm1d(in_size))
                    
            layers.append(nn.Linear(in_size, out_size))

            activation_function = getattr(nn, disc_prior_activation_function)()  
            layers.append(activation_function)

            if disc_prior_use_layer_normalization:
                layers.append(nn.LayerNorm())
            
            if disc_prior_dropout_prob > 0.0:
                layers.append(nn.Dropout(p=disc_prior_dropout_prob))
                
            in_size = out_size
        
        layers.append(nn.Linear(in_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        yhat = self.model(x)
        return yhat

disc_prior = DiscriminatorPrior().to(device)
    
print(disc_prior)

x = torch.rand((batch_size, latent_dim), dtype=torch.float32).to(device)
x.shape

x = disc_prior(x)
x.shape

"""
for name, param in discriminator_prior.named_parameters():
    print(f"Layer: {name} | Size: {param.size()}")
"""

if save_models == True:
    
    # save using pickle
    torch.save(disc_prior, "{}.pth".format(disc_prior_model_file))
    
    # save using onnx
    x = torch.zeros((1, latent_dim)).to(device)
    torch.onnx.export(disc_prior, x, "{}.onnx".format(disc_prior_model_file))

if save_tscript == True:
    
    disc_prior.eval()
    
    # save using TochScript
    #x = torch.rand((batch_size, latent_dim), dtype=torch.float32).to(device)
    x = torch.rand((1, latent_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(disc_prior, x)
    script_module.save("{}.pt".format(disc_prior_model_file))
    
    disc_prior.train()

if load_weights and disc_prior_weights_file:
    disc_prior.load_state_dict(torch.load(disc_prior_weights_file))

# Dilation Block

class DilationBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, stride, dilation, activation):
        super().__init__()
        
        branch1 = []
        branch1.append(nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2))
        
        if activation != None:
            activation_function = getattr(nn, disc_prior_activation_function)() 
            branch1.append(activation_function)
        
        self.branch1 = nn.Sequential(*branch1)
        
        branch2 = []
        branch2.append(nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2))
        
        if activation != None:
            activation_function = getattr(nn, disc_prior_activation_function)() 
            branch2.append(activation_function)
        
        branch2.append(nn.Conv1d(in_channels=out_filters, out_channels=out_filters, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2))
        
        if activation != None:
            activation_function = getattr(nn, disc_prior_activation_function)() 
            branch2.append(activation_function)    
        
        self.branch2 = nn.Sequential(*branch2)

    def forward(self, x):
        
        #print("dilation x ", x.shape)
        
        x1 = self.branch1(x)
        
        #print("dilation x1 ", x1.shape)
        
        x2 = self.branch2(x)
        
        #print("dilation x2 ", x2.shape)
        
        x = x1 + x2
        
        #print("dilation x3 ", x.shape)
        
        return x

"""
# debug

in_size = 256
in_filters = 32
out_filters = 32

x = torch.rand((batch_size, in_filters, in_size), dtype=torch.float32)
print("x1 ", x.shape)

kernel_size = 7
stride = 4
dilation = 32
padding = (kernel_size - 1) // 2 * dilation

x = DilationBlock(in_filters=in_filters, out_filters=out_filters, kernel_size=kernel_size, stride=stride, dilation=dilation, activation="LeakyReLU")(x)
print("x2 ", x.shape)
"""

# Transposed Dilation Block

class TransposedDilationBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, stride, dilation, activation):
        super().__init__()
        
        activation_function = getattr(nn, disc_prior_activation_function)() 
        
        branch1 = []
        branch1.append(nn.ConvTranspose1d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, output_padding=(kernel_size - 1) // 2))
        
        if activation != None:
            activation_function = getattr(nn, disc_prior_activation_function)() 
            branch1.append(activation_function)
        
        self.branch1 = nn.Sequential(*branch1)
        
        branch2 = []
        branch2.append(nn.ConvTranspose1d(in_channels=in_filters, out_channels=in_filters, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, output_padding=(kernel_size - 1) // 2))
        
        if activation != None:
            activation_function = getattr(nn, disc_prior_activation_function)() 
            branch2.append(activation_function) 
            
        branch2.append(nn.ConvTranspose1d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2))
        
        if activation != None:
            activation_function = getattr(nn, disc_prior_activation_function)() 
            branch2.append(activation_function)
 
        self.branch2 = nn.Sequential(*branch2)

    def forward(self, x):
        
        #print("dilation x ", x.shape)
        
        x1 = self.branch1(x)
        
        #print("dilation x1 ", x1.shape)
        
        x2 = self.branch2(x)
        
        #print("dilation x2 ", x2.shape)
        
        x = x1 + x2
        
        #print("dilation x3 ", x.shape)
        
        return x    

"""
# debug

in_size = 256
in_filters = 32
out_filters = 32

x = torch.rand((batch_size, in_filters, in_size), dtype=torch.float32)
print("x1 ", x.shape)

kernel_size = 7
stride = 4
dilation = 32
padding = (kernel_size - 1) // 2 * dilation

x = TransposedDilationBlock(in_filters=in_filters, out_filters=out_filters, kernel_size=kernel_size, stride=stride, dilation=dilation, activation="LeakyReLU")(x)
print("x2 ", x.shape)
"""
        
# Create Audio Discriminator

class DiscriminatorAudio(nn.Module):
    def __init__(self):
        super(DiscriminatorAudio, self).__init__()
        
        self.conv_model = None
        self.dense_model = None
        
        activation = disc_audio_activation_function
        
        conv_layer_count = len(disc_audio_conv_filter_counts)
        dense_layer_count = len(disc_audio_dense_layer_sizes)
        
        in_size = audio_window_length
        in_channels = 1
        
        if conv_layer_count > 0:
            
            layers = []
            
            for layer_index in range(conv_layer_count):
                
                out_channels = disc_audio_conv_filter_counts[layer_index]
                kernel_size = disc_audio_conv_kernel_sizes[layer_index]
                stride = disc_audio_conv_strides[layer_index]
                dilation = disc_audio_conv_dilations[layer_index]
                
                if disc_audio_use_batch_normalization:
                    layers.append(nn.BatchNorm1d(in_channels))
                    
                if dilation > 0:
                    layers.append(DilationBlock(in_channels, out_channels, kernel_size, stride, dilation, activation ))
                else:
                    layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2))
                    activation_function = getattr(nn, activation)()  
                    layers.append(activation_function)
    
                if disc_audio_use_layer_normalization:
                    layers.append(nn.LayerNorm())
                
                if disc_audio_dropout_prob > 0.0:
                    layers.append(nn.Dropout(p=disc_prior_dropout_prob))
                
                #print("conv ", layer_index, " size_i ", in_size, " size_o ", (in_size // stride), " channels_i ", in_channels, " channels_o ", out_channels  )
                
                in_channels = out_channels
                in_size = in_size // stride # only correct for padding == "same"
        
            self.conv_model = nn.Sequential(*layers)
            
            # flattened size
            in_size *= out_channels
        
        if dense_layer_count > 0:
            
            layers = []
            
            for layer_index in range(dense_layer_count):
                
                out_size = disc_audio_dense_layer_sizes[layer_index]

                if disc_audio_use_batch_normalization:
                    layers.append(nn.BatchNorm1d(in_size))
                
                layers.append(nn.Linear(in_size, out_size))
                
                activation_function = getattr(nn, activation)()  
                layers.append(activation_function)
                
                if disc_audio_use_layer_normalization:
                    layers.append(nn.LayerNorm())
                
                if disc_audio_dropout_prob > 0.0:
                    layers.append(nn.Dropout(p=disc_prior_dropout_prob))
                    
                #print("dense ", layer_index, " size_i ", in_size, " size_o ", out_size )
                   
                in_size = out_size
        
        if disc_audio_use_batch_normalization:
            layers.append(nn.BatchNorm1d(in_size))
            
        layers.append(nn.Linear(in_size, 1))
        layers.append(nn.Sigmoid())

        self.dense_model = nn.Sequential(*layers)

    def forward(self, x):
        
        #print("x1 ", x.shape)
        
        if self.conv_model != None:
            x = torch.unsqueeze(x, dim=1) 
            
            #print("x2 ", x.shape)
            
            x = self.conv_model(x)
            
            #print("x3 ", x.shape)
        
        x = x.view(-1, x.shape[1] * x.shape[2])
        
        #print("x4 ", x.shape)
        
        yhat = self.dense_model(x)
        
        #print("x5 ", yhat.shape)
        
        return yhat

disc_audio = DiscriminatorAudio().to(device)
    
print(disc_audio)

x = torch.rand((batch_size, audio_window_length), dtype=torch.float32).to(device)
x.shape

x = disc_audio(x)
x.shape

"""
for name, param in discriminator_prior.named_parameters():
    print(f"Layer: {name} | Size: {param.size()}")
"""

if save_models == True:
    # save using pickle
    torch.save(disc_audio, "{}.pth".format(disc_audio_model_file))
    
    # save using onnx
    x = torch.zeros((1, audio_window_length)).to(device)
    torch.onnx.export(disc_audio, x, "{}.onnx".format(disc_audio_model_file))

if save_tscript == True:
    
    disc_audio.eval()
    
    # save using TochScript
    x = torch.rand((1, audio_window_length), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(disc_audio, x)
    script_module.save("{}.pt".format(disc_audio_model_file))
    
    disc_audio.train()

if load_weights and disc_audio_weights_file:
    disc_audio.load_state_dict(torch.load(disc_audio_weights_file))


# Create Audio Encoder

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        
        self.conv_model = None
        self.dense_model = None
        
        activation = ae_activation_function
        
        conv_layer_count = len(ae_conv_filter_counts)
        dense_layer_count = len(ae_dense_layer_sizes)
        
        in_size = audio_window_length
        in_channels = 1
        
        if conv_layer_count > 0:

            layers = []
            
            for layer_index in range(conv_layer_count):
                
                out_channels = ae_conv_filter_counts[layer_index]
                kernel_size = ae_conv_kernel_sizes[layer_index]
                stride = ae_conv_strides[layer_index]
                dilation = ae_conv_dilations[layer_index]
                
                if ae_use_batch_normalization:
                    layers.append(nn.BatchNorm1d(in_channels))
                    
                if dilation > 0:
                    layers.append(DilationBlock(in_channels, out_channels, kernel_size, stride, dilation, activation ))
                else:
                    layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2))
                    activation_function = getattr(nn, activation)()  
                    layers.append(activation_function)
    
                if ae_use_layer_normalization:
                    layers.append(nn.LayerNorm())
                
                if ae_dropout_prob > 0.0:
                    layers.append(nn.Dropout(p=ae_dropout_prob))
                
                #print("conv ", layer_index, " size_i ", in_size, " size_o ", (in_size // stride), " channels_i ", in_channels, " channels_o ", out_channels  )
                
                in_channels = out_channels
                in_size = in_size // stride # only correct for padding == "same"
        
            self.conv_model = nn.Sequential(*layers)
            
            # flattened size
            in_size *= out_channels
            
        if dense_layer_count > 0:
            
            layers = []
            
            for layer_index in range(dense_layer_count):
                
                out_size = ae_dense_layer_sizes[layer_index]

                if disc_audio_use_batch_normalization:
                    layers.append(nn.BatchNorm1d(in_size))
                
                layers.append(nn.Linear(in_size, out_size))
                
                activation_function = getattr(nn, activation)()  
                layers.append(activation_function)
                
                if ae_use_layer_normalization:
                    layers.append(nn.LayerNorm())
                
                if ae_dropout_prob > 0.0:
                    layers.append(nn.Dropout(p=ae_dropout_prob))
                    
                #print("dense ", layer_index, " size_i ", in_size, " size_o ", out_size )
                   
                in_size = out_size
        
        if ae_use_batch_normalization:
            layers.append(nn.BatchNorm1d(in_size))
            
        layers.append(nn.Linear(in_size, latent_dim))

        self.dense_model = nn.Sequential(*layers)

    def forward(self, x):
        
        #print("AudioEncoder forward")
        
        #print("x1 ", x.shape)
        
        if self.conv_model != None:
            x = torch.unsqueeze(x, dim=1) 
            
            #print("x2 ", x.shape)
            
            x = self.conv_model(x)
            
            #print("x3 ", x.shape)
        
        x = x.view(-1, x.shape[1] * x.shape[2])
        
        #print("x4 ", x.shape)
        
        yhat = self.dense_model(x)
        
        #print("x5 ", yhat.shape)
        
        return yhat
    
ae_encoder = AudioEncoder().to(device)
    
print(ae_encoder)

x = torch.rand((batch_size, audio_window_length), dtype=torch.float32).to(device)
x.shape

x = ae_encoder(x)
x.shape

"""
for name, param in discriminator_prior.named_parameters():
    print(f"Layer: {name} | Size: {param.size()}")
"""

if save_models == True:
    # save using pickle
    torch.save(ae_encoder, "{}.pth".format(ae_encoder_model_file))
    
    # save using onnx
    x = torch.zeros((1, audio_window_length)).to(device)
    torch.onnx.export(ae_encoder, x, "{}.onnx".format(ae_encoder_model_file))

if save_tscript == True:
    
    ae_encoder.eval()
    
    # save using TochScript
    x = torch.rand((1, audio_window_length), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(ae_encoder, x)
    script_module.save("{}.pt".format(ae_encoder_model_file))
    
    ae_encoder.train()

if load_weights and ae_encoder_weights_file:
    ae_encoder.load_state_dict(torch.load(ae_encoder_weights_file))   

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

x = torch.rand((batch_size, 1, audio_window_length), dtype=torch.float32).to(device)
x = ae_encoder.conv_model(x)
shape_before_flattening = list(x.shape)
shape_after_flattening = [shape_before_flattening[0], shape_before_flattening[1] * shape_before_flattening[2]]

class AudioDecoder(nn.Module):
    def __init__(self):
        super(AudioDecoder, self).__init__()
        
        self.dense_model = None
        self.conv_model = None
        
        activation = ae_activation_function
        
        dense_layer_count = len(rev_ae_dense_layer_sizes)
        conv_layer_count = len(ae_conv_filter_counts)
        
        in_size = latent_dim
        
        if dense_layer_count > 0:
            
            layers = []
            
            for layer_index in range(dense_layer_count):
                
                out_size = rev_ae_dense_layer_sizes[layer_index]

                if disc_audio_use_batch_normalization:
                    layers.append(nn.BatchNorm1d(in_size))
                
                layers.append(nn.Linear(in_size, out_size))
                
                activation_function = getattr(nn, activation)()  
                layers.append(activation_function)
                
                if disc_audio_use_layer_normalization:
                    layers.append(nn.LayerNorm())
                
                if disc_audio_dropout_prob > 0.0:
                    layers.append(nn.Dropout(p=disc_prior_dropout_prob))
                    
                #print("dense ", layer_index, " size_i ", in_size, " size_o ", out_size )
                   
                in_size = out_size
            
            if ae_use_batch_normalization:
                layers.append(nn.BatchNorm1d(in_size))
                
            out_size = shape_after_flattening[1]
            layers.append(nn.Linear(in_size, out_size))
            
            in_size = out_size
                
            if conv_layer_count > 0:
                activation_function = getattr(nn, activation)()  
                layers.append(activation_function)

            if ae_use_layer_normalization:
                layers.append(nn.LayerNorm())
            
            self.dense_model = nn.Sequential(*layers)
        
        if conv_layer_count > 0:

            layers = []
            
            in_channels = in_size
            in_size = 1
            
            for layer_index in range(conv_layer_count - 1):
                
                out_channels = rev_ae_conv_filter_counts[layer_index]
                kernel_size = rev_ae_conv_kernel_sizes[layer_index]
                stride = rev_ae_conv_strides[layer_index]
                dilation = rev_ae_conv_dilations[layer_index]
                
                if ae_use_batch_normalization:
                    layers.append(nn.BatchNorm1d(in_channels))
                    
                if dilation > 0:
                    layers.append(TransposedDilationBlock(in_channels, out_channels, kernel_size, stride, dilation, activation ))
                else:
                    layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, output_padding=(kernel_size - 1) // 2))
                    activation_function = getattr(nn, activation)()  
                    layers.append(activation_function)
    
                if ae_use_layer_normalization:
                    layers.append(nn.LayerNorm())
                
                if ae_dropout_prob > 0.0:
                    layers.append(nn.Dropout(p=ae_dropout_prob))
                
                #print("conv ", layer_index, " size_i ", in_size, " size_o ", in_size * stride, " channels_i ", in_channels, " channels_o ", out_channels, " kernel ", kernel_size, " stride ", stride, " dilation ", dilation  )

                in_channels = out_channels
                in_size = in_size * stride # only correct for padding == "same"
            
            
            out_channels = rev_ae_conv_filter_counts[-1]
            kernel_size = rev_ae_conv_kernel_sizes[-1]
            stride = rev_ae_conv_strides[-1]
            dilation = rev_ae_conv_dilations[-1]
            
            if ae_use_batch_normalization:
                layers.append(nn.BatchNorm1d(in_channels))
                
            if dilation > 0:
                layers.append(TransposedDilationBlock(in_channels, out_channels, kernel_size, stride, dilation, None ))
            else:
                layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, output_padding=(kernel_size - 1) // 2))

            #print("conv ", conv_layer_count - 1, " size_i ", in_size, " size_o ", in_size * stride, " channels_i ", in_channels, " channels_o ", out_channels, " kernel ", kernel_size, " stride ", stride, " dilation ", dilation  )

            self.conv_model = nn.Sequential(*layers)
    
    def forward(self, x):
        
        #print("AudioDecoder forward")
        
        #print("x1 ", x.shape)
        
        x = self.dense_model(x)
            
        #print("x2 ", x.shape)
            
        if self.conv_model != None:
            
            #x = x.view(shape_before_flattening)
            x = x.view([x.shape[0]] + shape_before_flattening[1:])
            
            #print("x2 ", x.shape)
            
            x = self.conv_model(x)
            
            #print("x3 ", x.shape)
        
        x = x.view(-1, x.shape[1] * x.shape[2])
        
        #print("x4 ", x.shape)
        
        yhat = x
        
        #print("x5 ", yhat.shape)
        
        return yhat
                

ae_decoder = AudioDecoder().to(device)
    
print(ae_decoder)

x = torch.rand((batch_size, latent_dim), dtype=torch.float32).to(device)
x.shape

x = ae_decoder(x)
x.shape

if save_models == True:
    # save using pickle
    torch.save(ae_decoder, "{}.pth".format(ae_decoder_model_file))
    
    # save using onnx
    x = torch.zeros((1, latent_dim)).to(device)
    torch.onnx.export(ae_decoder, x, "{}.onnx".format(ae_decoder_model_file))

if save_tscript == True:
    
    ae_decoder.eval()
    
    # save using TochScript
    x = torch.rand((1, latent_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(ae_decoder, x)
    script_module.save("{}.pt".format(ae_decoder_model_file))
    
    ae_decoder.train()

if load_weights and ae_decoder_weights_file:
    ae_decoder.load_state_dict(torch.load(ae_decoder_weights_file))   

# Training

# loss functions

cross_entropy = nn.BCELoss()

# discriminator prior loss
def disc_prior_loss(d_x, g_z, smoothing_factor = 0.9):
    
    ones = torch.ones_like(d_x).to(device)
    zeros = torch.zeros_like(g_z).to(device)
    
    real_loss = cross_entropy(d_x, ones * smoothing_factor) # If we feed the discriminator with real images, we assume they all are the right pictures --> Because of that label == 1
    fake_loss = cross_entropy(g_z, zeros) # Each noise we feed in are fakes image --> Because of that labels are 0. 
    total_loss = real_loss + fake_loss
    
    return total_loss

# discriminator audio loss
def disc_audio_loss(d_x, g_z, smoothing_factor = 0.9):
    
    ones = torch.ones_like(d_x).to(device)
    zeros = torch.zeros_like(g_z).to(device)
    
    real_loss = cross_entropy(d_x, ones * smoothing_factor) # If we feed the discriminator with real images, we assume they all are the right pictures --> Because of that label == 1
    fake_loss = cross_entropy(g_z, zeros) # Each noise we feed in are fakes image --> Because of that labels are 0. 
    total_loss = real_loss + fake_loss
    
    return total_loss

# autoencoder audio reconstruction loss
def ae_rec_loss(y, yhat):
    flat_y = torch.flatten(y)
    flat_yhat = torch.flatten(yhat)
    _loss = torch.mean((flat_y-flat_yhat)**2)
    
    return _loss

# autoencoder disc prior loss
def ae_disc_prior_loss(dx_of_gx):
    ones = torch.ones_like(dx_of_gx).to(device)
    return cross_entropy(dx_of_gx, ones) 

# autoencoder disc audio loss
def ae_disc_audio_loss(dx_of_gx):
    ones = torch.ones_like(dx_of_gx).to(device)
    return cross_entropy(dx_of_gx, ones) 

# autoencoder l1 loss
def ae_l1_loss(z):
    flat_z = torch.flatten(z)
    _loss = torch.mean(torch.abs(flat_z))
    
    return _loss

# autoencoder l2 loss
def ae_l2_loss(z):
    flat_z = torch.flatten(z)
    _loss = torch.mean(flat_z**2)
    
    return _loss

# optimizers

disc_prior_optimizer = torch.optim.Adam(disc_prior.parameters(), lr=disc_prior_learning_rate)
disc_audio_optimizer = torch.optim.Adam(disc_audio.parameters(), lr=disc_audio_learning_rate)
ae_optimizer = torch.optim.Adam(list(ae_encoder.parameters()) + list(ae_decoder.parameters()), lr=ae_learning_rate)

# Train Step

def train_step(ae_encoder, ae_decoder, disc_prior, disc_audio, audio_batch, batch_size = 32):
    
    # train disc_prior
    with torch.no_grad():
        fake_normal = ae_encoder(audio_batch)
    
    real_normal = sample_normal(fake_normal.shape)
    
    disc_prior_real_output =  disc_prior(real_normal)
    disc_prior_fake_output =  disc_prior(fake_normal)   
    
    #print("disc_prior_real_output ",disc_prior_real_output)
    #print("disc_prior_fake_output ",disc_prior_fake_output)
    
    _disc_prior_loss = disc_prior_loss(disc_prior_real_output, disc_prior_fake_output)

    # Backpropagation
    disc_prior_optimizer.zero_grad()
    _disc_prior_loss.backward()
    disc_prior_optimizer.step()

    # train disc_audio
    with torch.no_grad():
        fake_audio = ae_decoder(ae_encoder(audio_batch))
        
    real_audio = audio_batch
    
    disc_audio_real_output =  disc_audio(real_audio)
    disc_audio_fake_output =  disc_audio(fake_audio)   
    
    #print("disc_audio_real_output ",disc_audio_real_output)
    #print("disc_audio_fake_output ",disc_audio_fake_output)
    
    _disc_audio_loss = disc_audio_loss(disc_audio_real_output, disc_audio_fake_output)

    # Backpropagation
    disc_audio_optimizer.zero_grad()
    _disc_audio_loss.backward()
    disc_audio_optimizer.step()

    # train autoencoder
    encoder_out = ae_encoder(audio_batch)
    decoder_out = ae_decoder(encoder_out)
    
    disc_prior_fake_output =  disc_prior(encoder_out)  
    disc_audio_fake_output =  disc_audio(decoder_out)
        
    _ae_rec_loss = ae_rec_loss(audio_batch, decoder_out) 
    _ae_disc_prior_loss = ae_disc_prior_loss(disc_prior_fake_output)
    _ae_disc_audio_loss = ae_disc_audio_loss(disc_audio_fake_output)
    _ae_l1_loss = ae_l1_loss(encoder_out)
    _ae_l2_loss = ae_l2_loss(encoder_out)
        
    _ae_loss = _ae_rec_loss * ae_rec_loss_scale + _ae_disc_prior_loss * ae_disc_prior_loss_scale + _ae_disc_audio_loss * ae_disc_audio_loss_scale + _ae_l1_loss * ae_l1_loss_scale + _ae_l2_loss * ae_l2_loss_scale

    #print("train_step _ae_loss ", _ae_loss.detach().cpu().numpy(), "_ae_rec_loss ", _ae_rec_loss.detach().cpu().numpy(), " _ae_disc_prior_loss ", _ae_disc_prior_loss.detach().cpu().numpy(), " _ae_disc_audio_loss ", _ae_disc_audio_loss.detach().cpu().numpy(), " _ae_l1_loss ", _ae_l1_loss.detach().cpu().numpy(), " _ae_l2_loss ", _ae_l2_loss.detach().cpu().numpy())


    # Backpropagation
    ae_optimizer.zero_grad()
    _ae_loss.backward()
    ae_optimizer.step()
        
    return _ae_loss, _disc_prior_loss, _disc_audio_loss, _ae_rec_loss, _ae_disc_prior_loss, _ae_disc_audio_loss, _ae_l1_loss, _ae_l2_loss
  
# Test Step
      
def test_step(ae_encoder, ae_decoder, disc_prior, disc_audio, audio_batch, batch_size = 32):
    
    # test disc_prior
    with torch.no_grad():
        fake_normal = ae_encoder(audio_batch)
        real_normal = sample_normal(fake_normal.shape)
    
        disc_prior_real_output =  disc_prior(real_normal)
        disc_prior_fake_output =  disc_prior(fake_normal)   
        _disc_prior_loss = disc_prior_loss(disc_prior_real_output, disc_prior_fake_output)

    # test disc_audio
    with torch.no_grad():
        fake_audio = ae_decoder(ae_encoder(audio_batch))
        real_audio = audio_batch
    
        disc_audio_real_output =  disc_audio(real_audio)
        disc_audio_fake_output =  disc_audio(fake_audio)   
        _disc_audio_loss = disc_audio_loss(disc_audio_real_output, disc_audio_fake_output)

    # test autoencoder
    with torch.no_grad():
        encoder_out = ae_encoder(audio_batch)  
        decoder_out = ae_decoder(encoder_out)
    
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
    loss_history["ae disc prior train loss"] = []
    loss_history["ae disc audio train loss"] = []
    loss_history["ae l1 train loss"] = []
    loss_history["ae l2 train loss"] = []
    loss_history["ae test loss"] = []
    loss_history["disc prior test loss"] = []
    loss_history["disc audio test loss"] = []
    loss_history["ae rec test loss"] = []
    loss_history["ae disc prior test loss"] = []
    loss_history["ae disc audio test loss"] = []
    loss_history["ae l1 test loss"] = []
    loss_history["ae l2 test loss"] = []

    for epoch in range(epoches):
        
        start = time.time()
        
        ae_train_loss_per_epoch = []
        disc_prior_train_loss_per_epoch = []
        disc_audio_train_loss_per_epoch = []
        ae_rec_train_loss_per_epoch = []
        ae_disc_prior_train_loss_per_epoch = []
        ae_disc_audio_train_loss_per_epoch = []
        ae_l1_train_loss_per_epoch = []
        ae_l2_train_loss_per_epoch = []
        
        for train_batch in train_dataset:
            
            train_batch = train_batch.to(device)
            
            _ae_loss, _disc_prior_loss, _disc_audio_loss, _ae_rec_loss, _ae_disc_prior_loss, _ae_disc_audio_loss, _ae_l1_loss, _ae_l2_loss = train_step(ae_encoder, ae_decoder, disc_prior, disc_audio, train_batch, batch_size = batch_size)

            _ae_loss = _ae_loss.detach().cpu().numpy()
            _disc_prior_loss = _disc_prior_loss.detach().cpu().numpy()
            _disc_audio_loss = _disc_audio_loss.detach().cpu().numpy() 
            _ae_rec_loss = _ae_rec_loss.detach().cpu().numpy()
            _ae_disc_prior_loss = _ae_disc_prior_loss.detach().cpu().numpy()
            _ae_disc_audio_loss = _ae_disc_audio_loss.detach().cpu().numpy() 
            _ae_l1_loss = _ae_l1_loss.detach().cpu().numpy()
            _ae_l2_loss = _ae_l2_loss.detach().cpu().numpy()
            
            #print("batch ae_loss ", _ae_loss, " disc_prior_loss ", _disc_prior_loss, " disc_audio_loss ", _disc_audio_loss, " ae_rec_loss ", _ae_rec_loss, " ae_disc_prior_loss ", _ae_disc_prior_loss, " ae_disc_audio_loss ", _ae_disc_audio_loss, " ae_l1_loss ", _ae_l1_loss, " ae_l2_loss ", _ae_l2_loss)

            ae_train_loss_per_epoch.append(_ae_loss)
            disc_prior_train_loss_per_epoch.append(_disc_prior_loss)
            disc_audio_train_loss_per_epoch.append(_disc_audio_loss)
            ae_rec_train_loss_per_epoch.append(_ae_rec_loss)
            ae_disc_prior_train_loss_per_epoch.append(_ae_disc_prior_loss)
            ae_disc_audio_train_loss_per_epoch.append(_ae_disc_audio_loss)
            ae_l1_train_loss_per_epoch.append(_ae_l1_loss)
            ae_l2_train_loss_per_epoch.append(_ae_l2_loss)

        ae_train_loss_per_epoch = np.mean(np.array(ae_train_loss_per_epoch))
        disc_prior_train_loss_per_epoch = np.mean(np.array(disc_prior_train_loss_per_epoch))
        disc_audio_train_loss_per_epoch = np.mean(np.array(disc_audio_train_loss_per_epoch))
        ae_rec_train_loss_per_epoch = np.mean(np.array(ae_rec_train_loss_per_epoch))
        ae_disc_prior_train_loss_per_epoch = np.mean(np.array(ae_disc_prior_train_loss_per_epoch))
        ae_disc_audio_train_loss_per_epoch = np.mean(np.array(ae_disc_audio_train_loss_per_epoch))
        ae_l1_train_loss_per_epoch = np.mean(np.array(ae_l1_train_loss_per_epoch))
        ae_l2_train_loss_per_epoch = np.mean(np.array(ae_l2_train_loss_per_epoch))
        
        """
        print("epoch ae_loss ", ae_train_loss_per_epoch, " disc_prior_loss ", disc_prior_train_loss_per_epoch, " disc_audio_loss ", disc_audio_train_loss_per_epoch, 
              " ae_rec_loss ", ae_rec_train_loss_per_epoch, " ae_disc_prior_loss ", ae_disc_prior_train_loss_per_epoch, " ae_disc_audio_loss ", ae_disc_audio_train_loss_per_epoch, 
              " ae_l1_loss ", ae_l1_train_loss_per_epoch, " ae_l2_loss ", ae_l2_train_loss_per_epoch)
        """
        
        ae_test_loss_per_epoch = []
        disc_prior_test_loss_per_epoch = []
        disc_audio_test_loss_per_epoch = []
        ae_rec_test_loss_per_epoch = []
        ae_disc_prior_test_loss_per_epoch = []
        ae_disc_audio_test_loss_per_epoch = []
        ae_l1_test_loss_per_epoch = []
        ae_l2_test_loss_per_epoch = []
        
        for test_batch in test_dataset:
            
            test_batch = test_batch.to(device)
            
            _ae_loss, _disc_prior_loss, _disc_audio_loss, _ae_rec_loss, _ae_disc_prior_loss, _ae_disc_audio_loss, _ae_l1_loss, _ae_l2_loss = test_step(ae_encoder, ae_decoder, disc_prior, disc_audio, test_batch, batch_size = batch_size)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            _disc_prior_loss = _disc_prior_loss.detach().cpu().numpy()
            _disc_audio_loss = _disc_audio_loss.detach().cpu().numpy()  
            _ae_rec_loss = _ae_rec_loss.detach().cpu().numpy()
            _ae_disc_prior_loss = _ae_disc_prior_loss.detach().cpu().numpy()
            _ae_disc_audio_loss = _ae_disc_audio_loss.detach().cpu().numpy()  
            _ae_l1_loss = _ae_l1_loss.detach().cpu().numpy()
            _ae_l2_loss = _ae_l2_loss.detach().cpu().numpy()

            ae_test_loss_per_epoch.append(_ae_loss)
            disc_prior_test_loss_per_epoch.append(_disc_prior_loss)
            disc_audio_test_loss_per_epoch.append(_disc_audio_loss)
            ae_rec_test_loss_per_epoch.append(_ae_rec_loss)
            ae_disc_prior_test_loss_per_epoch.append(_ae_disc_prior_loss)
            ae_disc_audio_test_loss_per_epoch.append(_ae_disc_audio_loss)
            ae_l1_test_loss_per_epoch.append(_ae_l1_loss)
            ae_l2_test_loss_per_epoch.append(_ae_l2_loss)
            
            
        ae_test_loss_per_epoch = np.mean(np.array(ae_test_loss_per_epoch))
        disc_prior_test_loss_per_epoch = np.mean(np.array(disc_prior_test_loss_per_epoch))
        disc_audio_test_loss_per_epoch = np.mean(np.array(disc_audio_test_loss_per_epoch))
        ae_rec_test_loss_per_epoch = np.mean(np.array(ae_rec_test_loss_per_epoch))
        ae_disc_prior_test_loss_per_epoch = np.mean(np.array(ae_disc_prior_test_loss_per_epoch))
        ae_disc_audio_test_loss_per_epoch = np.mean(np.array(ae_disc_audio_test_loss_per_epoch))
        ae_l1_test_loss_per_epoch = np.mean(np.array(ae_l1_test_loss_per_epoch))
        ae_l2_test_loss_per_epoch = np.mean(np.array(ae_l2_test_loss_per_epoch))

        if epoch % model_save_interval == 0 and save_weights == True:
            disc_prior.save_weights("aae/weights/disc_prior_weights epoch_{}".format(epoch))
            disc_audio.save_weights("aae/weights/disc_audio_weights epoch_{}".format(epoch))
            ae_encoder.save_weights("aae/weights/ae_encoder_weights epoch_{}".format(epoch))
            ae_decoder.save_weights("aae/weights/ae_decoder_weights epoch_{}".format(epoch))
        
        if epoch % audio_save_interval == 0 and save_audio == True:
            create_epoch_sonifications(epoch)
            
        loss_history["ae train_loss"].append(ae_train_loss_per_epoch)
        loss_history["disc prior train loss"].append(disc_prior_train_loss_per_epoch)
        loss_history["disc audio train loss"].append(disc_audio_train_loss_per_epoch)
        loss_history["ae rec train loss"].append(ae_rec_train_loss_per_epoch)
        loss_history["ae disc prior train loss"].append(ae_disc_prior_train_loss_per_epoch)
        loss_history["ae disc audio train loss"].append(ae_disc_audio_train_loss_per_epoch)
        loss_history["ae l1 train loss"].append(ae_l1_train_loss_per_epoch)
        loss_history["ae l2 train loss"].append(ae_l2_train_loss_per_epoch)
        loss_history["ae test loss"].append(ae_test_loss_per_epoch)
        loss_history["disc prior test loss"].append(disc_prior_test_loss_per_epoch)
        loss_history["disc audio test loss"].append(disc_audio_test_loss_per_epoch)
        loss_history["ae rec test loss"].append(ae_rec_test_loss_per_epoch)
        loss_history["ae disc prior test loss"].append(ae_disc_prior_test_loss_per_epoch)
        loss_history["ae disc audio test loss"].append(ae_disc_audio_test_loss_per_epoch)
        loss_history["ae l1 test loss"].append(ae_l1_test_loss_per_epoch)
        loss_history["ae l2 test loss"].append(ae_l2_test_loss_per_epoch)

        print ('epoch {} :  ae train {:01.4f} dprior {:01.4f} daudio {:01.4f} dprior2 {:01.4f} daudio2 {:01.4f} rec {:01.4f} l1 {:01.4f} l2 {:01.4f} test {:01.4f} dprior {:01.4f} daudio {:01.4f} rec {:01.4f} l1 {:01.4f} l2 {:01.4f} time {:01.2f}'.format(epoch + 1, ae_train_loss_per_epoch, ae_disc_prior_train_loss_per_epoch, ae_disc_audio_train_loss_per_epoch, disc_prior_train_loss_per_epoch, disc_audio_train_loss_per_epoch, 
               ae_rec_train_loss_per_epoch, ae_l1_train_loss_per_epoch, ae_l2_train_loss_per_epoch, ae_test_loss_per_epoch, ae_disc_prior_test_loss_per_epoch, ae_disc_audio_test_loss_per_epoch, 
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
        
        target_audio = torch.from_numpy(target_audio).to(device)
        target_audio = torch.unsqueeze(target_audio, 0)
        
        with torch.no_grad():
            enc_audio = ae_encoder(target_audio)
            pred_audio = ae_decoder(enc_audio)
            
        pred_audio = np.squeeze(pred_audio.detach().cpu().numpy())
    
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
        seq2_audio_window = audio_standardized[target2_audio_start:target2_audio_end]
        seq1_audio_window = np.expand_dims(seq1_audio_window, axis=0)
        seq2_audio_window = np.expand_dims(seq2_audio_window, axis=0)
        seq1_audio_window = torch.from_numpy(seq1_audio_window).to(device)
        seq2_audio_window = torch.from_numpy(seq2_audio_window).to(device)
        
        with torch.no_grad():
            seq1_latent_vector = ae_encoder(seq1_audio_window)
            seq2_latent_vector = ae_encoder(seq2_audio_window)
        
        alpha = start_alpha + (end_alpha - start_alpha) * i / (audio_window_count - 1)
        mix_latent_vector = seq1_latent_vector * (1.0 - alpha) + seq2_latent_vector * alpha
        
        with torch.no_grad():
            pred_audio = ae_decoder(mix_latent_vector)
        
        pred_audio = (torch.squeeze(pred_audio)).detach().cpu().numpy()
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

        target_start_audio = torch.from_numpy(target_start_audio).to(device)
        target_end_audio = torch.from_numpy(target_end_audio).to(device)

        with torch.no_grad():
            start_enc = ae_encoder(target_start_audio)
            end_enc = ae_encoder(target_end_audio)

        for i in range(interpolation_window_count):
            inter_enc = start_enc + (end_enc - start_enc) * i / (interpolation_window_count - 1.0)
            
            with torch.no_grad():
                pred_audio = ae_decoder(inter_enc)
            
            pred_audio = (torch.squeeze(pred_audio)).detach().cpu().numpy()
            
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
    
    # put models into evaluation mode, turning off dropout and batchnorm
    disc_prior.eval()
    disc_audio.eval()
    ae_encoder.eval()
    ae_decoder.eval()
    
    # pred sonifications
    for audio_start_time in audio_save_start_times:
        
        file_name = "{}/pred_audio_{}_epoch_{}.wav".format(audio_result_path, audio_start_time, epoch)
        create_pred_sonification(audio_start_time, audio_save_duration, file_name)

    # two audio region interpolation
    for time_index in range(len(audio_save_start_times) - 1):
        audio_start_time1 = audio_save_start_times[time_index]
        audio_start_time2 = audio_save_start_times[time_index + 1]
        file_name = "{}/interpol_audio_{}_{}_epoch_{}.wav".format(audio_result_path, audio_start_time1, audio_start_time2, epoch)
        create_interpol_sonification(audio_start_time1, audio_start_time2, audio_save_duration, file_name)

    # audio window sequence traversal
    file_name = "{}/traverse_audio_{}_{}_{}_{}_epoch_{}.wav".format(audio_result_path, audio_traverse_start_window, audio_traverse_end_window, audio_traverse_window_count, audio_traverse_interpolation_count, epoch)
    create_traverse_sonifcation(audio_traverse_start_window, audio_traverse_end_window, audio_traverse_window_count, audio_traverse_interpolation_count, file_name)

    # put models back into training mode
    disc_prior.train()
    disc_audio.train()
    ae_encoder.train()
    ae_decoder.train()
 
# Train Model
loss_history = train(train_dataloader, test_dataloader, epochs)

# save history
if save_history:
    utils.save_loss_as_csv(loss_history, "../results_pytorch/sonification/aae/history.csv")
    utils.save_loss_as_image(loss_history, "../results_pytorch/sonification/aae/history.png")

# final weights save
if save_weights:
    torch.save(disc_prior.state_dict(), "../results_pytorch/sonification/aae/disc_prior_weights_epoch_{}".format(epochs))
    torch.save(disc_audio.state_dict(), "../results_pytorch/sonification/aae/disc_audio_weights_epoch_{}".format(epochs))
    torch.save(ae_encoder.state_dict(), "../results_pytorch/sonification/aae/ae_encoder_weights_epoch_{}".format(epochs))
    torch.save(ae_decoder.state_dict(), "../results_pytorch/sonification/aae/ae_decoder_weights_epoch_{}".format(epochs))

if save_audio:
    create_epoch_sonifications(epochs)

