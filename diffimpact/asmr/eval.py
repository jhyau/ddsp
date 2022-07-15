import sys, os
sys.path.append('..')
sys.path.append('.')
sys.path.append("../..")

import yaml
import cdpam
import argparse

from absl import logging
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

# For evaluation metrics
import scipy
from scipy.signal import hilbert
import librosa
import torch


#import wandb
#resume = False
#wandb_id = 'placeholder'
logging.set_verbosity(logging.INFO)

gin.external_configurable(tf.keras.regularizers.L1, module='tf.keras.regularizers')

"""
Two main sections:
    1. Running inference on DiffImpact checkpoints to get predicted audio
    2. Calculating the metrics on the audio
"""


# Load in impact profile predictions
def generate_audio_impact(predictions, modal_fir, reverb, impact, mags, taus, stdevs,
                   noise, acceleration_scale, revc, audio_sample_rate, example_secs, output_dir, scratch='controls',
                   modal_fir_only=False):
    """Generate DiffImpact's estimate of impact sound based on current model variables."""
    # Generate impulse --> impact profile
    # magnitude_envelopes, taus, prediction['stdevs']
#     impc = impact.get_controls(mags, stdevs, taus, 0) # needs to be 2D
#     print("tau_bias: ", predictions['tau_bias'])
    impc_gt = impact.get_controls(predictions['magnitudes'], predictions['stdevs'], predictions['taus'], 0)#predictions['tau_bias'])
    impulse_profile_gt = impact.get_signal(impc_gt['magnitudes'], impc_gt['taus'])

#     mags = tf.zeros([1, 2520,1]) # test passing in all zeros for mags
#     taus = tf.zeros([1, 2520, 1])

    impc = impact.get_controls(mags, stdevs, taus, 0)#predictions['tau_bias'])
    impulse_profile_pred = impact.get_signal(impc['magnitudes'], impc['taus'])

    print("GT impact shape: ", impulse_profile_gt.shape)
    print("pred impact shape: ", impulse_profile_pred.shape)

    print("model's impact profile (has some noise) num nonzeros: ", tf.math.count_nonzero(tf.math.round(predictions['impact']['signal'])))
    print("Generated impact profile from model's output num nonzero: ", tf.math.count_nonzero(tf.math.round(impulse_profile_gt)))
    print("Predicted impact profile from loading num nonzero: ", tf.math.count_nonzero(tf.math.round(impulse_profile_pred)))
    print("difference avg: ", tf.math.reduce_mean(impulse_profile_gt - impulse_profile_pred))

    # Check that the impulse profile generated is same as model output
#     check = tf.math.equal(impulse_profile, impulse_profile_scratch)
#     print(f"compared elements of impulse profile: {check}")
#     if (not tf.reduce_all(check) == True):
#         diff = impulse_profile - impulse_profile_scratch
#         print("average difference: ", tf.math.reduce_mean(diff))
#         print("original impulse profile: ", impulse_profile)
# #     assert(tf.reduce_all(check) == True)
#     print(f"element-wise comparison of network output impulse profile and from scratch: {tf.reduce_all(check)}")
#     print("impulse profile shape: ", impulse_profile_scratch.shape) # force profile

    # Generate modal FIR --> modal response (object material sound)
    # TODO: To use "raw" normally again, make sure to switch this!!
    irc_scratch = modal_fir.get_controls(predictions['gains'], predictions['frequencies'], predictions['dampings'])
    ir_scratch = modal_fir.get_signal(irc_scratch['gains'], irc_scratch['frequencies'], irc_scratch['dampings'])
    #ir = modal_fir.get_signal(irc['gains'], irc['frequencies'], irc['dampings'])# Modal response
#     ir = modal_fir.get_signal(gains, frequencies, dampings)
    modal_response = predictions['modal_fir']['signal']

    # Plotting the network output modal response
    print("Plotting convolve generated modal response")
#     output_dir = f'/juno/u/jyau/asmr-video-to-sound/ddsp/diffimpact/asmr/ckpt/multiclass-1hr/modal_fir'
    os.makedirs(output_dir, exist_ok=True)
    begin_offset = int(train_sample_rate * 1.95)
    cutoff = int(train_sample_rate * 2.5)

    ir_plot = np.squeeze(ir_scratch) # modal response
    print(ir_plot.shape)
    ir_plot = ir_plot[begin_offset:cutoff]
    t2 = (np.arange(0, ir_plot.shape[0])  - (train_sample_rate * 2 - begin_offset)) / (2 * train_sample_rate)
    plt.plot(t2, ir_plot)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    fig = plt.gcf()
    fig.set_size_inches(fig_width+2, fig_height)
    if no_axis:
        ax = plt.gca()
        # ax.axes.yaxis.set_visible(False)
#         ax.axes.yaxis.set_ticks([])
        plt.xticks(fontsize=font_size-6)
    fig.savefig(os.path.join(output_dir, audio_title+f'_convolve_ir-material_id_{material_id}.png'), bbox_inches='tight')
    plt.show()

    check = tf.math.equal(modal_response, ir_scratch)
    print(f"compared elements of modal response: {check}")
    assert(tf.reduce_all(check) == True)
    print(f"element-wise comparison of network output modal response and from scratch: {tf.reduce_all(check)}")

    # Convolve together for modal vibration sounds
    if scratch == 'raw':
        audio = ddsp.core.fft_convolve(impulse_profile_pred, ir_scratch)
    else:
        audio = ddsp.core.fft_convolve(impulse_profile_pred, modal_response)
    print("convolved shape: ", audio.shape)

    if modal_fir_only:
        print("Return audio without adding noise, acceleration sound, or reverb")
        audio = ddsp.core.resample(audio, int(audio_sample_rate)*example_secs, 'linear')
        return audio, impulse_profile_pred

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
    return audio, impulse_profile_pred



# Generating audio from convolving each of the parts outputted by DiffImpact inference
def generate_audio(predictions, modal_fir, reverb, impact, impulse_profile, gains, frequencies, dampings, modal_response,
                   noise, acceleration_scale, revc, audio_sample_rate, example_secs, output_dir, scratch='controls',
                   modal_fir_only=False):
    """Generate DiffImpact's estimate of impact sound based on current model variables."""
    # Generate impulse --> impact profile
    # magnitude_envelopes, taus, prediction['stdevs']
#     impc = impact.get_controls(mags, stdevs, taus, 0) # needs to be 2D
#     print("tau_bias: ", predictions['tau_bias'])
    impc = impact.get_controls(predictions['magnitudes'], predictions['stdevs'], predictions['taus'], 0)#predictions['tau_bias'])
    impulse_profile_scratch = impact.get_signal(impc['magnitudes'], impc['taus'])

    # Check that the impulse profile generated is same as model output
    check = tf.math.equal(impulse_profile, impulse_profile_scratch)
    print(f"compared elements of modal response: {check}")
    if (not tf.reduce_all(check) == True):
        diff = impulse_profile - impulse_profile_scratch
        print("average difference: ", tf.math.reduce_mean(diff))
        print("original impulse profile: ", impulse_profile)
#     assert(tf.reduce_all(check) == True)
    print(f"element-wise comparison of network output impulse profile and from scratch: {tf.reduce_all(check)}")
    print("impulse profile shape: ", impulse_profile_scratch.shape) # force profile

    # Generate modal FIR --> modal response (object material sound)
    # TODO: To use "raw" normally again, make sure to switch this!!
    irc_scratch = modal_fir.get_controls(predictions['gains'], predictions['frequencies'], predictions['dampings'])
    ir_scratch = modal_fir.get_signal(irc_scratch['gains'], irc_scratch['frequencies'], irc_scratch['dampings'])
    #ir = modal_fir.get_signal(irc['gains'], irc['frequencies'], irc['dampings'])# Modal response
    ir = modal_fir.get_signal(gains, frequencies, dampings)

    # Plotting the network output modal response
    print("Plotting convolve generated modal response")
#     output_dir = f'/juno/u/jyau/asmr-video-to-sound/ddsp/diffimpact/asmr/ckpt/multiclass-1hr/modal_fir'
    os.makedirs(output_dir, exist_ok=True)
    begin_offset = int(train_sample_rate * 1.95)
    cutoff = int(train_sample_rate * 2.5)

    ir_plot = np.squeeze(ir_scratch) # modal response
    print(ir_plot.shape)
    ir_plot = ir_plot[begin_offset:cutoff]
    t2 = (np.arange(0, ir_plot.shape[0])  - (train_sample_rate * 2 - begin_offset)) / (2 * train_sample_rate)
    plt.plot(t2, ir_plot)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    fig = plt.gcf()
    fig.set_size_inches(fig_width+2, fig_height)
    if no_axis:
        ax = plt.gca()
        # ax.axes.yaxis.set_visible(False)
#         ax.axes.yaxis.set_ticks([])
        plt.xticks(fontsize=font_size-6)
    fig.savefig(os.path.join(output_dir, audio_title+f'_convolve_ir-material_id_{material_id}.png'), bbox_inches='tight')
    plt.show()

    print("model's output modal response: ", modal_response)
    print("ir_scratch: ", ir_scratch)
    print("ir: ", ir)

    check = tf.math.equal(modal_response, ir_scratch)
    print(f"compared elements of modal response: {check}")
    assert(tf.reduce_all(check) == True)
    print(f"element-wise comparison of network output modal response and from scratch: {tf.reduce_all(check)}")

    # Convolve together for modal vibration sounds
    if scratch == 'raw':
        audio = ddsp.core.fft_convolve(impulse_profile_scratch, ir_scratch)
    elif scratch == 'controls':
        audio = ddsp.core.fft_convolve(impulse_profile_scratch, ir)
    else:
        audio = ddsp.core.fft_convolve(impulse_profile_scratch, modal_response)
    print("convolved shape: ", audio.shape)

    if modal_fir_only:
        print("Return audio without adding noise, acceleration sound, or reverb")
        audio = ddsp.core.resample(audio, int(audio_sample_rate)*example_secs, 'linear')
        return audio

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





"""
Functions used for evaluation: metrics, normalization, etc.
"""
def high_pass_filter(clip, fs):
    """
    fs is sampling frequency (audio sampling rate)
    """
    sos = scipy.signal.butter(10, 50, 'hp', fs=fs, output='sos')
    clip = scipy.signal.sosfiltfilt(sos, clip)
    return clip


def Envelope_distance(predicted_binaural, gt_binaural):
    #channel1
    pred_env_channel1 = np.abs(hilbert(predicted_binaural[0,:]))
    gt_env_channel1 = np.abs(hilbert(gt_binaural[0,:]))
    channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))

    #channel2
#     pred_env_channel2 = np.abs(hilbert(predicted_binaural[1,:]))
#     gt_env_channel2 = np.abs(hilbert(gt_binaural[1,:]))
#     channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2)**2))

    #sum the distance between two channels
    envelope_distance = channel1_distance #+ channel2_distance
    return float(envelope_distance)

def STFT_L2_distance(predicted_binaural, gt_binaural):
    #channel1
    predicted_spect_channel1 = librosa.core.stft(np.asfortranarray(predicted_binaural[0,:]), n_fft=512, hop_length=160, win_length=400, center=True)
    gt_spect_channel1 = librosa.core.stft(np.asfortranarray(gt_binaural[0,:]), n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(predicted_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel1), axis=0)
    predicted_realimag_channel1 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel1), axis=0)
    gt_realimag_channel1 = np.concatenate((real, imag), axis=0)
    channel1_distance = np.mean(np.power((predicted_realimag_channel1 - gt_realimag_channel1), 2))

    #channel2
#     predicted_spect_channel2 = librosa.core.stft(np.asfortranarray(predicted_binaural[1,:]), n_fft=512, hop_length=160, win_length=400, center=True)
#     gt_spect_channel2 = librosa.core.stft(np.asfortranarray(gt_binaural[1,:]), n_fft=512, hop_length=160, win_length=400, center=True)
#     real = np.expand_dims(np.real(predicted_spect_channel2), axis=0)
#     imag = np.expand_dims(np.imag(predicted_spect_channel2), axis=0)
#     predicted_realimag_channel2 = np.concatenate((real, imag), axis=0)
#     real = np.expand_dims(np.real(gt_spect_channel2), axis=0)
#     imag = np.expand_dims(np.imag(gt_spect_channel2), axis=0)
#     gt_realimag_channel2 = np.concatenate((real, imag), axis=0)
#     channel2_distance = np.mean(np.power((predicted_realimag_channel2 - gt_realimag_channel2), 2))

    #sum the distance between two channels
    stft_l2_distance = channel1_distance #+ channel2_distance
    return float(stft_l2_distance)


def normalize_amp(input_audio, max_amp=32767.0, normalize=True):
    if not normalize:
        return input_audio
    
    result = np.zeros_like(input_audio)
    
    # Find the max amplitude in audio
    max_peak = np.max(np.abs(input_audio))
    ratio = max_amp / max_peak #(equivalent to 1 / (25000 / 32767.0))
    print("Max peak: ", max_peak)
    
    for i in range(input_audio.shape[0]): # iterate through the samples
        result[i,:] = np.round(input_audio[i,:] * ratio)
    return result

def pad_audio(audio, shape):
    result = np.zeros(shape)
    result[:audio.shape[0], :audio.shape[1]] = audio
    return result

"""
Running metrics calculations and comparisons for the diffimpact ckpt predictions
"""
def metrics_calculations(gt, pred, outfile, loss_fn, gt_title, material_id):
    # Calculate CDPAM loss
    dist = loss_fn.forward(gt, pred)
    loss = dist.cpu().detach().numpy()
    print(f"CDPAM loss between {gt_title} and diffimpact synth material id {material_id} of pred: ", loss)
    outfile.write(f"CDPAM loss between {gt_title} and diffimpact synth material id {material_id} of pred: {loss}\n")

    # L2 loss (mean squared loss) comparison
    l2loss = tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(gt, pred)))
    l2norm = tf.norm(gt - pred)
    print(f"L2 loss between {gt_title} and diffimpact synth material id {material_id} of pred: {l2loss} also {l2norm}")
    outfile.write(f"L2 loss between {gt_title} and diffimpact synth material id {material_id} of pred: {l2loss} also {l2norm}\n")

    # L1 loss comparison
    l1loss = tf.reduce_sum(tf.math.abs(gt - pred))
    l1norm = tf.norm(gt - pred, ord=1)
    print(f"L1 loss between {gt_title} and diffimpact synth material id {material_id} of pred: {l1loss} and {l1norm}")
    outfile.write(f"L1 loss between {gt_title} and diffimpact synth material id {material_id} of pred: {l1loss} and {l1norm}\n")

    # Envelope distance
    env = Envelope_distance(pred, gt)
    print(f"Envelope distance between {gt_title} and diffimpact synth material id {material_id} of pred: {env}")
    outfile.write(f"Envelope distance between {gt_title} and diffimpact synth material id {material_id} of pred: {env}\n")

    # STFT L2 distance
    stft_l2 = STFT_L2_distance(pred, gt)
    print(f"STFT L2 distance between {gt_title} and diffimpact synth material id {material_id} of pred: {stft_l2}")
    outfile.write(f"STFT L2 distance between {gt_title} and diffimpact synth material id {material_id} of pred: {stft_l2}\n")


def metrics(args, diffimpact_ckpt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initializing CDPAM loss
    loss_fn = cdpam.CDPAM(dev=device)

    # Getting material dictionary
    material_dict = get_material_dict(args.diffimpact_ckpt_path)

    # Iterate through the validation audio examples to calculate metrics
    audio_list = get_eval_examples_list(args)
    with open(os.path.join(args.save_dir, "metrics.txt"), "w") as outfile:
        for aud in audio_list:
            material_id = determine_material_id(args, aud, material_dict)

            # Load audios
            orig_gt = cdpam.load_audio(f"/juno/u/jyau/asmr-video-to-sound/data/asmr_all/valid/{aud}.wav")
            gt = cdpam.load_audio(f'/juno/u/jyau/asmr-video-to-sound/data/asmr_all/high_pass/valid/{aud}.wav')
            repeated_gt = cdpam.load_audio(f"/juno/u/jyau/regnet/data/features/tapping/materials/audio_10s/{aud}.wav")[:, :220500]
            diff_pred = cdpam.load_audio(os.path.join(args.save_dir, aud+"_"+args.input_type+"_"+diffimpact_ckpt+"_material_id_"+str(material_id)+".wav"))

            # Normalize amplitudes
            gt_max_amp = np.max(np.abs(gt))
            orig_gt_max_amp = np.max(np.abs(orig_gt))
            repeated_gt_max_amp = np.max(np.abs(repeated_gt))
            print(f"Max amplitude in ground truth with high pass filter: {gt_max_amp} and max amp in original gt: {orig_gt_max_amp} and max amp in repeated gt: {repeated_gt_max_amp}")
            print("gt shape: ", gt.shape)
            gt = normalize_amp(gt)
            orig_gt = normalize_amp(orig_gt)
            repeated_gt = normalize_amp(repeated_gt)
            diff_pred = normalize_amp(diff_pred)

            print("========================Evaluation on audio: ", aud)
            print("************************Compare against high pass filter ground truth...")

            outfile.write(f"Calculating metrics for {aud} using material id: {material_id}\n")

            # If the shape is different due to needing repeat
            if gt.shape != diff_pred.shape:
                outfile.write("The original audio was <10 seconds, so this one needed repeating\n")
                print(f"orig gt shape: {orig_gt.shape}, high pass gt shape: {gt.shape}, repated_gt shape: {repeated_gt.shape}, diffimpact pred shape: {diff_pred.shape}")
                metrics_calculations(repeated_gt, diff_pred, outfile, loss_fn, "repeated gt", material_id)
                continue

            metrics_calculations(gt, diff_pred, outfile, loss_fn, "high pass gt", material_id)
            print("*********************Compare against original gt....")
            metrics_calculations(orig_gt, diff_pred, outfile, loss_fn, "orig gt (no high pass filter)", material_id)
            
            outfile.write("====================================================================================\n")
    print("Finished calculating metrics for diffimpact predictions")

"""
Running DiffImpact inference
"""
def get_material_dict(diffimpact_ckpt_path):
    # Get material labels from diffimpact ckpt
    with open(os.path.join(diffimpact_ckpt_path, "material_id_table.txt"), "r") as file:
        label_dict = {}
        for line in file:
            tokens = line.split(" ")
            label_dict[tokens[0].strip()] = int(tokens[-1].strip())
        print(f"Number of keys in material labels dict: {len(label_dict)}")
    return label_dict


def get_eval_examples_list(args):
    # Getting list of audio/mel spec(s) to pass as input
    if args.audio_title:
        audio_input_list = [args.audio_title]
    else:
        audio_input_list = []
        # Get all audio from validation set
        with open(args.valid_spec_file, 'r') as outfile:
            for line in outfile:
                aud_name = line.strip()
                audio_input_list.append(aud_name)
    print("Number of examples to evaluate: ", len(audio_input_list))
    return audio_input_list


def determine_material_id(args, video_name, material_dict):
    # Determining material ID to use
    if args.material_id:
        material_id = args.material_id
    else:
        tokens = video_name.split("-")
        key = tokens[0] + "-" + tokens[1]
        if key not in material_dict:
            raise Exception("missing key for labels!!")
        material_id = material_dict[key]
    return material_id


# Running inference with DiffImpact checkpoint
def inference(args):
    example_secs = 10
    offset_secs = 0
    
    # Loading in diffimpact config file and getting useful parameters
    latest_operative_config = ddsp.training.train_util.get_latest_operative_config(args.diffimpact_ckpt_path)
    gin.parse_config_file(latest_operative_config)
    print("Latest operative config used: ", latest_operative_config)

    n_modal_freq = gin.config.query_parameter('%N_MODAL_FREQUENCIES')
    print(f"n modal frequencies: {n_modal_freq}")

    print("Original internal audio samples: ", gin.config.query_parameter("%INTERNAL_AUDIO_SAMPLES"))
    train_sample_rate = gin.config.query_parameter('%AUDIO_SAMPLE_RATE')
    train_samples = gin.config.query_parameter('%N_AUDIO_SAMPLES')

    # Preparing the configs for running inference on test/validation examples
    train_z_steps = gin.config.query_parameter('MfccTimeDistributedRnnEncoder.z_time_steps')
    offset_samples = int(offset_secs * train_sample_rate)
    test_samples = int(example_secs * train_sample_rate)
    test_z_steps = int(example_secs / (train_samples / train_sample_rate) * train_z_steps)
    gin.config.bind_parameter('%N_AUDIO_SAMPLES', test_samples)

    # Change the MFCC mel samples if using waveglow calculation
    new_mel_samples = example_secs * 172
    gin.config.bind_parameter('MfccTimeDistributedRnnEncoder.mel_samples', new_mel_samples)
    print("mel samples: ", new_mel_samples)

    # Force profile prediction noise
    gin.config.bind_parameter("Impact.include_noise", args.no_include_noise)
    print("Impact synth module including noise? ", gin.config.query_parameter("Impact.include_noise"))

    # Preparing the correct configs
    gin.config.bind_parameter('FilteredNoiseExpDecayReverb.gain_initial_bias', -4)
    gin.config.bind_parameter('FilteredNoiseExpDecayReverb.decay_initial_bias', 4.0)
    gin.config.bind_parameter('MfccTimeDistributedRnnEncoder.z_time_steps', test_z_steps)

    print(gin.config.query_parameter('FilteredNoiseExpDecayReverb.gain_initial_bias'))
    print(gin.config.query_parameter('FilteredNoiseExpDecayReverb.decay_initial_bias'))
    print(gin.config.query_parameter('%VALIDATION_FILE_PATTERN'))
    print("test z time steps: ", gin.config.query_parameter('MfccTimeDistributedRnnEncoder.z_time_steps'))
    print("test internal audio samples: ", gin.config.query_parameter("%INTERNAL_AUDIO_SAMPLES"))

    # Use Regnet predicted mel spectrogram as input, if want to pass spectrograms as input
    if args.input_type == "spec":
        gin.config.bind_parameter('MfccTimeDistributedRnnEncoder.mel_bins', 80)
        gin.config.bind_parameter('MfccTimeDistributedRnnEncoder.spectral_fn', gin.config.parse_value("@ddsp.spectral_ops.compute_mfcc_mel_spec"))
    
    print(gin.config.query_parameter('MfccTimeDistributedRnnEncoder.spectral_fn'))
    print(gin.config.query_parameter('MfccTimeDistributedRnnEncoder.mel_bins'))


    # Loading the diffimpact checkpoint
    model = ddsp.training.models.get_model()
    model.restore(args.diffimpact_ckpt_path)

    # Getting the DiffImpact ckpt name
    diffimpact_ckpt_tokens = args.diffimpact_ckpt_path.split("/")
    no_empty = [x for x in diffimpact_ckpt_tokens if x]
    diffimpact_ckpt = no_empty[-1]
    print("DiffImpact checkpoint name: ", diffimpact_ckpt)

    # Getting list of audio/mel spec(s) to pass as input
    audio_input_list = get_eval_examples_list(args)

    # Getting material dictionary
    material_dict = get_material_dict(args.diffimpact_ckpt_path)

    # Create save dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Reading in the audio or mel spectrogram and running inference
    for aud in audio_input_list:
        # Get material id to use
        material_id = determine_material_id(args, aud, material_dict)

        if args.input_type == "spec":
            spec = np.load(os.path.join(args.input_path, aud+".npy"))
            spec = tf.expand_dims(tf.convert_to_tensor(spec), axis=0)
            test_input = tf.data.Dataset.from_tensor_slices({'audio': spec, 'mel_spec': spec, 'material_id':[material_id], 'video_id':[0]}).batch(2)
        elif args.input_type == "audio":
            audio = tf.io.read_file(os.path.join(args.input_path, aud+".wav"))
            decoded_audio_init, audio_sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
            decoded_audio = tf.expand_dims(tf.squeeze(decoded_audio_init[offset_samples:(offset_samples + test_samples)]), axis=0)
            test_input = tf.data.Dataset.from_tensor_slices({'audio': decoded_audio, 'mel_spec': decoded_audio, 'material_id':[material_id], 'video_id':[0]}).batch(2)
        else:
            raise Exception("Unknown input type")

        # Run inference through diffimpact ckpt
        prediction = model(next(iter(test_input)), training=False)
        
        # Save the synthesized audio
        audio_path = os.path.join(args.save_dir, aud+"_"+args.input_type+"_"+diffimpact_ckpt+"_material_id_"+str(material_id)+".wav")
        transpose_aud = np.transpose(prediction['audio_synth'][:1, :].numpy())
        scipy.io.wavfile.write(audio_path, train_sample_rate, np.array(32767 * transpose_aud / np.max(np.abs(prediction['audio_synth']))).astype('int16'))

        # Using convolving to generate the audio waveform
    print("Finished with diffimpact inferences")

    print("Running metrics calculation for diffimpact ckpt outputs")
    metrics(args, diffimpact_ckpt)


if __name__ == '__main__':
    # Read in parameters needed for evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffimpact_ckpt_path", type=str, help="Path to the diffimpact checkpoint to use")
    parser.add_argument("--save_dir", type=str, help="Path to save the diffimpact inference outputs and plots")
    parser.add_argument("--input_type", type=str, default="spec", help="Type of input to pass into DiffImpact inference. Default is mel spectrogram (spec)")
    parser.add_argument("--input_path", type=str, help="Path to retrieve the inputs to diffimpact")
    parser.add_argument("--valid_spec_file", type=str, help="File to all validation/test set examples (basically multiple audios to infer as once)")
    parser.add_argument("--audio_title", type=str, default=None, help="Specify a specific audio to pass in. Leave blank to use all validation audio")
    parser.add_argument("--material_id", type=int, default=None, help="If want to use a specific material ID to retrieve a modal response. Otherwise, leave blank to use the corresponding ID")
    parser.add_argument("--no_include_noise", action="store_false", help="Use this flag to not include noise in force profile prediction")
    args = parser.parse_args()

    # Run evaluation/inference
    inference(args)
