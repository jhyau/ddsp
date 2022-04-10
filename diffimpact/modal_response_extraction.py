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
parser.add_argument("--train_pattern", type=str, default=None, help="glob pattern for train files")
parser.add_argument("--valid_pattern", type=str, default=None, help="glob pattern for valid files")
args = parser.parse_args()
print(args)

os.makedirs(args.output_path, exist_ok=True)

# Save the final differential components used to generate the audio
other_path = os.path.join(args.output_path, "ddsp_inference_final_outputs")
os.makedirs(other_path, exist_ok=True)

# Save the force profile components (before Impact module)
impulse_components_path = os.path.join(args.output_path, "impulse_profile_components")
os.makedirs(impulse_components_path, exist_ok=True)

# Save the modal response components (before ModalFIR module)
modal_components_path = os.path.join(args.output_path, "modal_response_components")
os.makedirs(modal_components_path, exist_ok=True)

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
if args.train_pattern is None:
    train_files = gin.config.query_parameter('%TRAIN_FILE_PATTERN')
else:
    train_files = args.train_pattern

if args.valid_pattern is None:    
    val_files = gin.config.query_parameter('%VALIDATION_FILE_PATTERN')
else:
    val_files = args.valid_pattern

print(f"Train files: {train_files}")
print(f"Valid files: {val_files}")

# Check if there are multiple glob patterns
if train_files.find("|") != -1:
    patterns = train_files.split('|')
    train = []
    for pat in patterns:
        file_paths_sub = glob.glob(pat)
        train.extend(file_paths_sub)
else:
    train = glob.glob(train_files)


if val_files.find("|") != -1:
    patterns = val_files.split('|')
    val = []
    for pat in patterns:
        file_paths_sub = glob.glob(pat)
        val.extend(file_paths_sub)
else:
    val = glob.glob(val_files)


total = train + val


# Initialize the modalFIR and Impact modules
print("audio sampling rate: ", train_sample_rate)
sample_factor = 2
modal_fir = ddsp.synths.ModalFIR(n_samples=int(sample_factor * train_sample_rate), sample_rate=int(sample_factor * train_sample_rate),
                            initial_bias=-1.5, hz_max=20000.0, freq_scale_fn=ddsp.core.frequencies_critical_bands, freq_scale='mel')

impact = ddsp.synths.Impact(sample_rate=int(sample_factor * train_sample_rate), n_samples=int(sample_factor * (example_secs * train_sample_rate)),
        max_impact_frequency=20, 
        mag_scale_fn=ddsp.core.exp_sigmoid, include_noise=False)



# Load model checkpoint
model = ddsp.training.models.get_model()
model.restore(args.save_path)

# Run inference to get the impact/force profile from each clip
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

    # Save the magnitudes, stdevs, and taus for impact profile
    mags = prediction['magnitudes']
    stdevs = prediction['stdevs']
    taus = prediction['taus']
    tau_bias = prediction['tau_bias']

    np.save(os.path.join(impulse_components_path, "_magnitudes.npy"), mags.numpy())
    np.save(os.path.join(impulse_components_path, "_taus.npy"), taus.numpy())
    np.save(os.path.join(impulse_components_path, "_stdevs.npy"), stdevs.numpy())
    np.save(os.path.join(impulse_components_path, "_tau_bias.npy"), tau_bias.numpy()) # Note that this is actually from the modal response pipeline
    
    # Save the impact profile without noise
    impc = impact.get_controls(prediction['magnitudes'], prediction['stdevs'], prediction['taus'], prediction['tau_bias'])
    impulse_profile_scratch = impact.get_signal(impc['magnitudes'], impc['taus'])
    np.save(os.path.join(other_path, name+"_no_noise_impulse_profile.npy"), impulse_profile_scratch.numpy())

        
    # For single ckpts, can also save modal response
    if not os.path.exists(os.path.join(args.save_path, "material_id_table.txt")):
        # Extract the raw gains, frequences, and dampings
        gains = prediction['gains']
        frequencies = prediction['frequencies']
        dampings = prediction['dampings']

        np.save(os.path.join(modal_components_path, name+"_gains_raw.npy"), gains.numpy())
        np.save(os.path.join(modal_components_path, name+"_freqs_raw.npy"), frequencies.numpy())
        np.save(os.path.join(modal_components_path, name+"_dampings_raw.npy"), dampings.numpy())

        # Get the gains, frequencies, and dampings after passing through scaling function
        control_gains = prediction['modal_fir']['controls']['gains']
        control_freqs = prediction['modal_fir']['controls']['frequencies']
        control_dampings = prediction['modal_fir']['controls']['dampings']

        np.save(os.path.join(modal_components_path, name+"_gains_controls.npy"), control_gains.numpy())
        np.save(os.path.join(modal_components_path, name+"_freqs_controls.npy"), control_freqs.numpy())
        np.save(os.path.join(modal_components_path, name+"_dampings_controls.npy"), control_dampings.numpy())


if not os.path.exists(os.path.join(args.save_path, "material_id_table.txt")):
    sys.exit(0)


# Separately save the modal responses for multiclass checkpoint
print("Saving modal responses...")

with open(os.path.join(args.save_path, "material_id_table.txt"), "r") as file:
    for line in file:
        video_name, material_id = line.split(' ')

        name = total[0].split('/')[-1].split('.')[0]
        print(f"Inferencing on: {clip} with material id: {material_id}")
        audio = tf.io.read_file(clip)
        decoded_audio, audio_sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
        decoded_audio = tf.expand_dims(tf.squeeze(decoded_audio[offset_samples:(offset_samples + test_samples)]), axis=0)
        test_input = tf.data.Dataset.from_tensor_slices({'audio':decoded_audio, 'material_id':[int(material_id)], 'video_id':[0]}).batch(2)

        # Have model checkpoint infer on the audio
        prediction = model(next(iter(test_input)), training=False)

        # Extract the raw gains, frequences, and dampings
        gains = prediction['gains']
        frequencies = prediction['frequencies']
        dampings = prediction['dampings']

        np.save(os.path.join(modal_components_path, name+f"_gains_raw-material_id-{material_id}.npy"), gains.numpy())
        np.save(os.path.join(modal_components_path, name+f"_freqs_raw-material_id-{material_id}.npy"), frequencies.numpy())
        np.save(os.path.join(modal_components_path, name+f"_dampings_raw-material_id-{material_id}.npy"), dampings.numpy())

        # Get the gains, frequencies, and dampings after passing through scaling function
        control_gains = prediction['modal_fir']['controls']['gains']
        control_freqs = prediction['modal_fir']['controls']['frequencies']
        control_dampings = prediction['modal_fir']['controls']['dampings']

        np.save(os.path.join(modal_components_path, name+f"_gains_controls-material_id-{material_id}.npy"), control_gains.numpy())
        np.save(os.path.join(modal_components_path, name+f"_freqs_controls-material_id-{material_id}.npy"), control_freqs.numpy())
        np.save(os.path.join(modal_components_path, name+f"_dampings_controls-material_id-{material_id}.npy"), control_dampings.numpy())

