import sys, os
sys.path.append("..")
sys.path.append("../..")
sys.path.append(".")

from spectral_ops import *
import numpy as np
import tensorflow as tf

wave_mel_spec = np.load("/juno/u/jyau/regnet/data/features/tapping/materials/melspec_10s_44100hz/sIgkTYTWPz8-004-0023_mel.npy")
aud = tf.io.read_file('/juno/u/jyau/regnet/data/features/tapping/materials/audio_10s_44100hz/sIgkTYTWPz8-004-0023.wav')

decoded_aud, sr = tf.audio.decode_wav(aud, desired_channels=1)

print("tf decoded audio input: \n", decoded_aud)

logmel = compute_waveglow_logmel(tf.squeeze(decoded_aud), sample_rate=44100, lo_hz=0.0, hi_hz=16000.0, mel_samples=1720)

print("saved waveglow mel: \n", wave_mel_spec)

print("tf version: \n", logmel)

print(tf.math.reduce_all(tf.experimental.numpy.allclose(wave_mel_spec, logmel, atol=1e-5)))

# Diffimpact original compute logmel
orig_logmel = compute_logmel(tf.squeeze(decoded_aud), sample_rate=44100, lo_hz=4.0, hi_hz=16000.0, fft_size=1024, bins=128, overlap=0.75, pad_end=True)
print("diffimpact original mel spec: \n", orig_logmel)
