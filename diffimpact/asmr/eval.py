import sys, os
sys.path.append('..')
sys.path.append('.')

import yaml

from absl import logging
import argparse
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

# Eval with saved impact profile tensors
def generate_audio_testing_impact(prediction, modal_fir, reverb, impulse_profile, modal_response,
                           noise, acceleration_scale, revc, audio_sample_rate, example_secs, scratch='raw'):
    """Generate DiffImpact's estimate of impact sound based on current model variables."""
    # Generate impulse --> impact profile
    # magnitude_envelopes, taus, prediction['stdevs']

    # Get force profile from saved tensors
    if scratch == "raw":
        impc = impact.get_controls(mags, stdevs, taus, 0) # needs to be 2D?
        impulse_profile = impact.get_signal(impc['magnitudes'], impc['taus'])
        
    print("impulse profile shape: ", impulse_profile.shape) # force profile

    # Get modal response from raw freqs, gains, and dampings
    irc_scratch = modal_fir.get_controls(raw_gain, raw_freq, raw_dampings)
    ir_scratch = modal_fir.get_signal(irc_scratch['gains'], irc_scratch['frequencies'], irc_scratch['dampings'])

    # Get modal response from scaled (passed through get_controls) freqs, gains, dampings
    ir = modal_fir.get_signal(gains, frequencies, dampings)

    print("ir: ", ir)
    print("model's output modal response: ", modal_response)
    #print("ir_scratch: ", ir_scratch)
    # Convolve together for modal vibration sounds
    #if scratch == 'raw':
    #    audio = ddsp.core.fft_convolve(impulse_profile, ir_scratch)
    #elif scratch == 'controls' or scratch =='control':
    #    audio = ddsp.core.fft_convolve(impulse_profile, ir)
    #else:
    audio = ddsp.core.fft_convolve(impulse_profile, modal_response) #Using modal response directly from Diffimpact's output
    print("convolved shape: ", audio.shape)

    # Generate and add time constant noise
    # Note that in the context, clips.shape[0] is batch size (which is 1 for all testing here)
    # clips.shape[1] is the actual clip size (like 441000 for 10 seconds of 44100 audio sampling rate)
    #unfiltered_noise = tf.random.uniform((clips.shape[0], int(clips.shape[1] * sample_factor)),
    #minval=-1.0, maxval=1.0)
    #noise = ddsp.core.frequency_filter(unfiltered_noise, ddsp.core.exp_sigmoid(noise_magnitudes - 4.0), 257)
    audio += noise
    print("after adding noise: ", audio.shape)

    # Add acceleration sound
    audio += impulse_profile * acceleration_scale
    print("after acceleration sound: ", audio.shape)

    # Add reverb
#     revc = reverb.get_controls(audio, reverb_gains, reverb_decay)
    audio = reverb.get_signal(audio, revc)#revc['ir'])
    print("after reverb: ", audio.shape)

    # Downsample from internal sampling rate to original recording sampling rate
    # audio = ddsp.core.resample(audio, clips.shape[1], 'linear')
    # Note that the resample function will return shape [n_timesteps], which is the second parameter
    print("audio sample rate: ", audio_sample_rate)
    audio = ddsp.core.resample(audio, int(audio_sample_rate)*example_secs, 'linear')
    return audio


def generate_audio(predictions, modal_fir, reverb, impulse_profile, gains, frequencies, dampings, modal_response,
                   noise, acceleration_scale, revc, audio_sample_rate, example_secs, scratch='controls'):
    """Generate DiffImpact's estimate of impact sound based on current model variables."""
    # Generate impulse --> impact profile
    # magnitude_envelopes, taus, prediction['stdevs']
#     impc = impact.get_controls(mags, stdevs, taus, 0) # needs to be 2D
#     impulse_profile = impact.get_signal(impc['magnitudes'], impc['taus'])
    print("impulse profile shape: ", impulse_profile.shape) # force profile

    # Generate modal FIR --> modal response (object material sound)
    irc_scratch = modal_fir.get_controls(predictions['gains'], predictions['frequencies'], predictions['dampings'])
    ir_scratch = modal_fir.get_signal(irc_scratch['gains'], irc_scratch['frequencies'], irc_scratch['dampings'])
    #ir = modal_fir.get_signal(irc['gains'], irc['frequencies'], irc['dampings'])# Modal response
    ir = modal_fir.get_signal(gains, frequencies, dampings)
    print("model's output modal response: ", modal_response)
    print("ir_scratch: ", ir_scratch)
    # Convolve together for modal vibration sounds
    if scratch == 'raw':
        audio = ddsp.core.fft_convolve(impulse_profile, ir_scratch)
    elif scratch == 'controls':
        audio = ddsp.core.fft_convolve(impulse_profile, ir)
    else:
        audio = ddsp.core.fft_convolve(impulse_profile, modal_response)
    print("convolved shape: ", audio.shape)

    # Generate and add time constant noise
    # Note that in the context, clips.shape[0] is batch size (which is 1 for all testing here)
    # clips.shape[1] is the actual clip size (like 441000 for 10 seconds of 44100 audio sampling rate)
    #unfiltered_noise = tf.random.uniform((clips.shape[0], int(clips.shape[1] * sample_factor)), minval=-1.0, maxval=1.0)
    #noise = ddsp.core.frequency_filter(unfiltered_noise, ddsp.core.exp_sigmoid(noise_magnitudes - 4.0), 257)
    audio += noise
    print("after adding noise: ", audio.shape)

    # Add acceleration sound
    audio += impulse_profile * acceleration_scale
    print("after acceleration sound: ", audio.shape)

    # Add reverb
#     revc = reverb.get_controls(audio, reverb_gains, reverb_decay)
    audio = reverb.get_signal(audio, revc)#revc['ir'])
    print("after reverb: ", audio.shape)

    # Downsample from internal sampling rate to original recording sampling rate
    # audio = ddsp.core.resample(audio, clips.shape[1], 'linear')
    # Note that the resample function will return shape [n_timesteps], which is the second parameter
    print("audio sample rate: ", audio_sample_rate)
    audio = ddsp.core.resample(audio, int(audio_sample_rate)*example_secs, 'linear')
    return audio


parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help="video name (do not include suffix like .npy)")
parser.add_argument('modal_response_path', type=str, help="path to where the modal responses are kept (where the ground truth gains, freqs, damps are)")
parser.add_argument('vocoder_type', type=str, help='vocoder material type')
parser.add_argument('vocoder_dir', type=str, help='path to the vocoder checkpoint')
parser.add_argument('eval_path', type=str, help='path to the outputted ddsp modal responses to convolve together')
parser.add_argument('--ddsp_inf_path', type=str, default=None, help='specify path to the ddsp inference outputs, if needed')
parser.add_argument('--example_secs', type=int, default=10)
parser.add_argument('--offset_secs', type=int, default=0)
args = parser.parse_args()


if not args.ddsp_inf_path:
    ddsp_inference_path = os.path.join(args.modal_response_path, 'ddsp_inference_outputs')
else:
    ddsp_inference_path = args.ddsp_inf_path

# TODO: clean this up from jupyter notebook style
# Evaluation by convolving the audio components (impulse/force profile, modal response, reverb, noise)
# Full pipeline
vocoder_type = "glass-bowl" # wood-box
save_dir = f'/juno/u/jyau/regnet/ddsp/diffimpact/asmr/regnet-labels/1hr-{vocoder_type}'
# save_dir = f'/juno/u/jyau/regnet/ddsp/diffimpact/checkpoints/{vocoder_type}'
example_secs = 10
offset_secs = 0

#latest_operative_config = ddsp.training.train_util.get_latest_operative_config(save_dir)
latest_operative_config = os.path.join(save_dir, 'operative_config-0.gin')
gin.parse_config_file(latest_operative_config)
print("Latest operative config used: ", latest_operative_config)

n_modal_freq = gin.config.query_parameter('%N_MODAL_FREQUENCIES')
print(f"n modal frequencies: {n_modal_freq}")
#gin.config.bind_parameter('%N_MODAL_FREQUENCIES', 64)
#n_modal_freq = gin.config.query_parameter('%N_MODAL_FREQUENCIES')
#print(f"n modal frequencies: {n_modal_freq}")

train_sample_rate = gin.config.query_parameter('%AUDIO_SAMPLE_RATE')
train_samples = gin.config.query_parameter('%N_AUDIO_SAMPLES')

audio_sample_rate = train_sample_rate

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
print(gin.config.query_parameter('FilteredNoiseExpDecayReverb.gain_initial_bias'))
print(gin.config.query_parameter('FilteredNoiseExpDecayReverb.decay_initial_bias'))
print(gin.config.query_parameter('%VALIDATION_FILE_PATTERN'))

# Initialize modules
sample_factor = 2
modal_fir = ddsp.synths.ModalFIR(n_samples=int(sample_factor * train_sample_rate),
                                 sample_rate=int(sample_factor * train_sample_rate),
                            initial_bias=-1.5, hz_max=20000.0, freq_scale_fn=ddsp.core.frequencies_critical_bands,
                                 freq_scale='mel')

impact = ddsp.synths.Impact(sample_rate=int(sample_factor * audio_sample_rate), n_samples=int(sample_factor),
                            max_impact_frequency=20, mag_scale_fn=ddsp.core.exp_sigmoid, include_noise=True)

reverb = ddsp.effects.FilteredNoiseExpDecayReverb(trainable=False, reverb_length=int(48000 * sample_factor),
                                                  decay_initial_bias=4.0, add_dry=True)

# Original ground truth
acceleration_scale = np.load(os.path.join(path, vid+"_acceleration_scale.npy"))
impulse_profile = np.load(os.path.join(path, vid+"_impulse_profile.npy"))
ir = np.load(os.path.join(path, vid+"_modal_response.npy")) # modal response
noise = np.load(os.path.join(path, vid+"_noise.npy"))
revc = np.load(os.path.join(path, vid+"_reverb.npy")) #np.concatenate((np.zeros(begin_offset), np.squeeze(prediction['reverb']['controls']['ir'])))
print(f"input shapes! impulse_profile: {impulse_profile.shape}, ir: {ir.shape}, noise: {noise.shape}, revc: {revc.shape}")

#raw_gains = np.expand_dims(np.load(os.path.join(eval_output, vid+".npy")), axis=1)
raw_gains = np.load(os.path.join(eval_output, vid+"_gain.npy"))
raw_gains_gt = np.load(os.path.join(freq_path, vid+"_gains_raw.npy"))
raw_freq = np.load(os.path.join(freq_path, vid+"_freqs_raw.npy"))
raw_dampings_gt = np.load(os.path.join(freq_path, vid+"_dampings_raw.npy"))
raw_dampings = np.load(os.path.join(eval_output, vid+"_dampings.npy"))

control_gains_gt = np.load(os.path.join(freq_path, vid+"_gains_controls.npy"))
control_freqs_gt = np.load(os.path.join(freq_path, vid+"_freqs_controls.npy"))
control_dampings_gt = np.load(os.path.join(freq_path, vid+"_dampings_controls.npy"))
print(f"raw gains gt shape: {raw_gains_gt.shape}")
print(f"raw freq gt shape: {raw_freq.shape}")
print(f"raw dampings gt shape: {raw_dampings.shape}")
# print(f"min raw gains :{np.min(control_freqs_gt)} and max: {np.max(control_freqs_gt)}")

# Use eval outputs (only raw gains)
#control_gains = np.load(os.path.join(eval_output, vid+".npy"))
#control_freqs = np.expand_dims(np.load(os.path.join(eval_output, vid+".npy")), axis=1)
#control_dampings = np.load(os.path.join(eval_output, vid+".npy"))
print(f"shapes: {raw_gains.shape}")
# print(f"min control freq:{np.min(control_freqs)} and max: {np.max(control_freqs)}")

audio_final = generate_audio_testing(raw_gains, raw_freq, raw_dampings, modal_fir, reverb, impulse_profile,
                             control_gains_gt, control_freqs_gt, control_dampings_gt,
                             ir, noise, acceleration_scale, revc, audio_sample_rate, example_secs, scratch='raw')
print(f"pred gains and dampings, final audio shape: {audio_final.shape}")
display(IPython.display.Audio(data=audio_final, rate=int(train_sample_rate)))

# If only using predicted damping
audio_final_damp = generate_audio_testing(raw_gains_gt, raw_freq, raw_dampings, modal_fir, reverb, impulse_profile,
                             control_gains_gt, control_freqs_gt, control_dampings_gt,
                             ir, noise, acceleration_scale, revc, audio_sample_rate, example_secs, scratch='raw')
print(f"pred dampings only audio shape: {audio_final_damp.shape}")
display(IPython.display.Audio(data=audio_final_damp, rate=int(train_sample_rate)))

# Only using predicted gains
audio_final_gain = generate_audio_testing(raw_gains, raw_freq, raw_dampings_gt, modal_fir, reverb, impulse_profile,
                             control_gains_gt, control_freqs_gt, control_dampings_gt,
                             ir, noise, acceleration_scale, revc, audio_sample_rate, example_secs, scratch='raw')
print(f"pred gains only audio shape: {audio_final_damp.shape}")
display(IPython.display.Audio(data=audio_final_gain, rate=int(train_sample_rate)))

# Ground truth comparison
audio_final_gt = generate_audio_testing(raw_gains_gt, raw_freq, raw_dampings_gt, modal_fir, reverb, impulse_profile,
                             control_gains_gt, control_freqs_gt, control_dampings_gt,
                             ir, noise, acceleration_scale, revc, audio_sample_rate, example_secs, scratch='raw')
print(f"gt final audio shape: {audio_final_gt.shape}")
display(IPython.display.Audio(data=audio_final_gt, rate=int(train_sample_rate)))





# Run audio through the DDSP checkpoint
try:
    audio = tf.io.read_file(f'/juno/u/jyau/regnet/data/features/ASMR/orig_asmr_by_material_clips/audio_10s_44100hz_ddsp/1hr/{vocoder_type}/train/{audio_title}.wav')
except Exception as ex:
    audio = tf.io.read_file(f'/juno/u/jyau/regnet/data/features/ASMR/orig_asmr_by_material_clips/audio_10s_44100hz_ddsp/3hr/{vocoder_type}/train/{audio_title}.wav')

print("audio shape: ", tf.shape(audio))
decoded_audio, audio_sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
print("decoded audio first shape: ", decoded_audio.shape)

#decoded_audio = decoded_audio * # Test by inputting zero vector audio
decoded_audio = tf.expand_dims(tf.squeeze(decoded_audio[offset_samples:(offset_samples + test_samples)]), axis=0)
#decoded_audio = tf.expand_dims(tf.squeeze(decoded_audio), axis=0)
print(decoded_audio.shape)
print("audio sample_rate: ", audio_sample_rate)
test_input = tf.data.Dataset.from_tensor_slices({'audio':decoded_audio, 'material_id':[0], 'video_id':[0]}).batch(2)
print("input to the model: ", test_input)
display(IPython.display.Audio(data=decoded_audio, rate=int(train_sample_rate)))

# audio of length of around 0.5 seconds (22050/256= 86 .1) for audio sampling rate 44100 and hop length 256
# Note that Sam's model uses mel_bins=128!
prediction = model(next(iter(test_input)), training=False)
print("pred audio shape: ", prediction['audio'].shape)
print("pred audio synth shape: ", prediction['audio_synth'].shape)
# ddsp.colab.colab_utils.specplot(prediction['audio'][:1, :]*MAX_WAV_VALUE, size=512)
ddsp.colab.colab_utils.specplot(prediction['audio'][:1, :], size=512)
ddsp.colab.colab_utils.specplot(prediction['audio_synth'][:1, :], size=512)
# ddsp.colab.colab_utils.specplot(prediction['audio'][:1, :train_sample_rate*2]*MAX_WAV_VALUE, size=512)
ddsp.colab.colab_utils.specplot(prediction['audio'][:1, :train_sample_rate*2], size=512)
ddsp.colab.colab_utils.specplot(prediction['audio_synth'][:1, :train_sample_rate*2], size=512)
plt.show()
display(IPython.display.Audio(data=prediction['audio_synth'][:1, :], rate=int(train_sample_rate)))
# IPython.display.Audio(data=prediction['modal_fir']['signal'][:1, :], rate=int(audio_sample_rate))

# Generate audio by replacing the frequency with eval
# Initialize modules
sample_factor = 2
modal_fir = ddsp.synths.ModalFIR(n_samples=int(sample_factor * train_sample_rate), sample_rate=int(sample_factor * train_sample_rate),
                            initial_bias=-1.5, hz_max=20000.0, freq_scale_fn=ddsp.core.frequencies_critical_bands, freq_scale='mel')
impact = ddsp.synths.Impact(sample_rate=int(sample_factor * audio_sample_rate), n_samples=int(sample_factor), max_impact_frequency=20, mag_scale_fn=ddsp.core.exp_sigmoid, include_noise=True)
reverb = ddsp.effects.FilteredNoiseExpDecayReverb(trainable=False, reverb_length=int(48000 * sample_factor), decay_initial_bias=4.0, add_dry=True)

acceleration_scale = prediction['acceleration_scalar']
impulse_profile = prediction['impact']['signal']
ir = prediction['modal_fir']['signal'] # modal response
noise = prediction['filtered_noise']['signal']
revc = prediction['reverb']['controls']['ir'] #np.concatenate((np.zeros(begin_offset), np.squeeze(prediction['reverb']['controls']['ir'])))
print(f"input shapes! impulse_profile: {impulse_profile.shape}, ir: {ir.shape}, noise: {noise.shape}, revc: {revc.shape}")
audio_final = generate_audio(prediction, modal_fir, reverb, impulse_profile,
                             prediction['modal_fir']['controls']['gains'],
                             prediction['modal_fir']['controls']['frequencies'],
                             prediction['modal_fir']['controls']['dampings'],
                             ir, noise, acceleration_scale, revc, audio_sample_rate, example_secs, scratch='raw')
print(f"final audio shape: {audio_final.shape} compared to audio_synth shape: {prediction['audio_synth'].shape}")
display(IPython.display.Audio(data=audio_final, rate=int(train_sample_rate)))
