import sys, os
sys.path.append('..')
sys.path.append('.')

import yaml
from absl import logging
import argparse
import glob
from tqdm import tqdm
import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tensorflow as tf
import ddsp.colab.colab_utils
from ddsp.spectral_ops import compute_mel, compute_logmel
import ddsp.training
import gin
import gin.tf
import gin.tf.external_configurables

#import wandb
#resume = False
#wandb_id = 'placeholder'
logging.set_verbosity(logging.INFO)

gin.external_configurable(tf.keras.regularizers.L1, module='tf.keras.regularizers')

# Set up argparser
parser = argparse.ArgumentParser("Generate the gains, frequencies, dampings, and force/impact profile given an audio clip and DDSP model checkpoint to load")
parser.add_argument('save_path', type=str, help="Path to the DDSP model's checkpoint directory")
parser.add_argument('gin_file', type=str, help="Path to the gin file to use")
parser.add_argument('output_path', type=str, help="Path to save the outputted files")
args = parser.parse_args()
print(args)

os.makedirs(args.output_path, exist_ok=True)
other_path = os.path.join(args.output_path, "ddsp_inference_outputs")
os.makedirs(other_path, exist_ok=True)

example_secs = 10
offset_secs = 0

# Load the gin file and parameters
gin.parse_config_file(args.gin_file)
train_sample_rate = gin.config.query_parameter('%AUDIO_SAMPLE_RATE')
train_samples = gin.config.query_parameter('%N_AUDIO_SAMPLES')

train_z_steps = gin.config.query_parameter('MfccTimeDistributedRnnEncoder.z_time_steps')

offset_samples = int(offset_secs * train_sample_rate)
test_samples = int(example_secs * train_sample_rate)

test_z_steps = int(example_secs / (train_samples / train_sample_rate) * train_z_steps)
gin.config.bind_parameter('%N_AUDIO_SAMPLES', test_samples)
try:
    train_internal_sample_rate = gin.config.query_parameter('%INTERNAL_SAMPLE_RATE')
    test_internal_samples = int(example_secs * train_internal_sample_rate)
    gin.config.bind_parameter('%INTERNAL_AUDIO_SAMPLES', test_internal_samples)
    gin.config.bind_parameter('FilteredNoise.initial_bias', gin.config.query_parameter('FilteredNoise.initial_bias') - 1.0)
except ValueError:
    pass
gin.config.bind_parameter('FilteredNoiseExpDecayReverb.gain_initial_bias', -4)
gin.config.bind_parameter('FilteredNoiseExpDecayReverb.decay_initial_bias', 4.0)
gin.config.bind_parameter('MfccTimeDistributedRnnEncoder.z_time_steps', test_z_steps)

# Get the trainig and validation files
train_files = gin.config.query_parameter('%TRAIN_FILE_PATTERN') 
val_files = gin.config.query_parameter('%VALIDATION_FILE_PATTERN')

train = glob.glob(train_files)
val = glob.glob(val_files)
total = train + val

# Load model checkpoint
model = ddsp.training.models.get_model()
model.restore(args.save_path)

# Run inference to get the gains, frequencies, dampings from before and after scaling function, and the impact/force profile from each clip
for clip in tqdm(total):
    name = clip.split('/')[-1].split('.')[0]
    print("Inferencing on: ", clip)
    audio = tf.io.read_file(clip)
    decoded_audio, audio_sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
    decoded_audio = tf.expand_dims(tf.squeeze(decoded_audio[offset_samples:(offset_samples + test_samples)]), axis=0)
    test_input = tf.data.Dataset.from_tensor_slices({'audio':decoded_audio, 'material_id':[0], 'video_id':[0]}).batch(2)

    # Have model checkpoint infer on the audio
    prediction = model(next(iter(test_input)), training=False)
    
    # Extract main outputs needed to generate audio
    acceleration_scale = prediction['acceleration_scalar']
    impulse_profile = prediction['impact']['signal']
    ir = prediction['modal_fir']['signal'] # modal response
    noise = prediction['filtered_noise']['signal']
    revc = prediction['reverb']['controls']['ir']
    
    # Save them
    np.save(os.path.join(other_path, name+"_acceleration_scale.npy"), acceleration_scale.numpy())
    np.save(os.path.join(other_path, name+"_impulse_profile.npy"), impulse_profile.numpy())
    np.save(os.path.join(other_path, name+"_modal_response.npy"), ir.numpy())
    np.save(os.path.join(other_path, name+"_noise.npy"), noise.numpy())
    np.save(os.path.join(other_path, name+"_reverb.npy"), revc.numpy())

    # Extract the raw gains, frequences, and dampings
    gains = prediction['gains']
    frequencies = prediction['frequencies']
    dampings = prediction['dampings']

    np.save(os.path.join(args.output_path, name+"_gains_raw.npy"), gains.numpy())
    np.save(os.path.join(args.output_path, name+"_freqs_raw.npy"), frequencies.numpy())
    np.save(os.path.join(args.output_path, name+"_dampings_raw.npy"), dampings.numpy())

    # Get the gains, frequencies, and dampings after passing through scaling function
    control_gains = prediction['modal_fir']['controls']['gains']
    control_freqs = prediction['modal_fir']['controls']['frequencies']
    control_dampings = prediction['modal_fir']['controls']['dampings']

    np.save(os.path.join(args.output_path, name+"_gains_controls.npy"), control_gains.numpy())
    np.save(os.path.join(args.output_path, name+"_freqs_controls.npy"), control_freqs.numpy())
    np.save(os.path.join(args.output_path, name+"_dampings_controls.npy"), control_dampings.numpy())

