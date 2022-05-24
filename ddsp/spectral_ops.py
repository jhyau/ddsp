# Copyright 2020 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Library of FFT operations for loss functions and conditioning."""

import crepe
from ddsp.core import safe_log
from ddsp.core import tf_float32
import gin
import librosa
import numpy as np
import tensorflow.compat.v2 as tf

from librosa.filters import mel as librosa_mel_fn
from scipy.signal import get_window
from librosa.util import pad_center

CREPE_SAMPLE_RATE = 16000
_CREPE_FRAME_SIZE = 1024

F0_RANGE = 127.0  # MIDI
LD_RANGE = 120.0  # dB

MAX_WAV_VALUE = 32768.0 # Waveglow mel spectrogram computation


def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
  """Differentiable stft in tensorflow, computed in batch."""
  assert frame_size * overlap % 2.0 == 0.0

  # Remove channel dim if present.
  audio = tf_float32(audio)
  if len(tf.shape(audio)) == 3:
    audio = tf.squeeze(audio, axis=-1)

  s = tf.signal.stft(
      signals=audio,
      frame_length=int(frame_size),
      frame_step=int(frame_size * (1.0 - overlap)),
      fft_length=int(frame_size),
      pad_end=pad_end)
  return s


def stft_np(audio, frame_size=2048, overlap=0.75, pad_end=True):
  """Non-differentiable stft using librosa, one example at a time."""
  assert frame_size * overlap % 2.0 == 0.0
  hop_size = int(frame_size * (1.0 - overlap))
  is_2d = (len(audio.shape) == 2)
  print("stft_np is_2d: ", is_2d)

  if pad_end:
    n_samples_initial = int(audio.shape[-1])
    n_frames = int(np.ceil(n_samples_initial / hop_size))
    n_samples_final = (n_frames - 1) * hop_size + frame_size
    pad = n_samples_final - n_samples_initial
    padding = ((0, 0), (0, pad)) if is_2d else ((0, pad),)
    audio = np.pad(audio, padding, 'constant')
    print("audio before stft shape: ", audio.shape)

  def stft_fn(y):
    return librosa.stft(y=y,
                        n_fft=int(frame_size),
                        hop_length=hop_size,
                        center=False).T

  s = np.stack([stft_fn(a) for a in audio]) if is_2d else stft_fn(audio)
  return s


@gin.register
def compute_mag(audio, size=2048, overlap=0.75, pad_end=True):
  mag = tf.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end))
  return tf_float32(mag)


@gin.register
def compute_mag_mel_spec(mel_spec, size=2048, overlap=0.75, pad_end=True):
  mag = tf.abs(mel_spec)
  return tf_float32(mag)


@gin.register
def compute_mel(audio,
                sample_rate=16000,
                lo_hz=0.0,
                hi_hz=8000.0,
                bins=64,
                fft_size=2048,
                overlap=0.75,
                pad_end=True):
  """Calculate Mel Spectrogram."""
  mag = compute_mag(audio, fft_size, overlap, pad_end)
  print("matrix of mag from compute_mel: ", mag)
  # mag shape: (1, time dim, num_bins)
  # num_spectrogram_bins = int(mag.shape[-1])
  num_spectrogram_bins = tf.cast(tf.shape(mag)[-1], tf.int32)
  if not bins:
    bins = int(fft_size / 4) + 1
  linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
      bins, num_spectrogram_bins, sample_rate, lo_hz, hi_hz)
  mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
  mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
  print("matrix of mel from compute_mel: ", mel)
  return mel

@gin.register
def compute_mel_from_mag(mag,
                         sample_rate=16000,
                         lo_hz=0.0,
                         hi_hz=8000.0,
                         bins=64,
                         fft_size=2048):
  num_spectrogram_bins = tf.cast(tf.shape(mag)[-1], tf.int32)
  if not bins:
    bins = int(fft_size / 4) + 1
  linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
      bins, num_spectrogram_bins, sample_rate, lo_hz, hi_hz)
  mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
  mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
  return mel

@gin.register
def compute_logmag(audio, size=2048, overlap=0.75, pad_end=True):
  return safe_log(compute_mag(audio, size, overlap, pad_end))


@gin.register
def compute_logmag_mel_spec(mel_spec, size=2048, overlap=0.75, pad_end=True):
  return safe_log(compute_mag_mel_spec(mel_spec, size, overlap, pad_end))



class TacotronSTFT(tf.keras.layers.Layer):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        #self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = tf.convert_to_tensor(mel_basis, dtype=tf.float32)
        #self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes, C=1, clip_val=1e-5):
        """
        PARAMS
        ------
        C: compression factor
        """
        return torch.log(tf.clip_by_value(magnitudes, clip_val, np.inf) * C)
        #return torch.log(torch.clamp(magnitudes, min=clip_val) * C)

    def spectral_de_normalize(self, magnitudes, C=1):
        """
        PARAMS
        ------
        C: compression factor used to compress
        """
        return tf.math.exp(x) / C

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(tf.math.reduce_min(y) >= -1)
        assert(tf.math.reduce_max(y) <= 1)

        magnitudes, phases = self.transform(y)
        magnitudes = magnitudes
        mel_output = tf.linalg.matmul(self.mel_basis, magnitudes) #torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

    def transform(self, input_data, window="hann"):
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[1]

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = tf.reshape(input_data, [num_batches, 1, num_samples])
        padding = tf.constant([[0, 0], [0, 0], [0, 0], [int(self.filter_length / 2), int(self.filter_length / 2)]])
        print(input_data.shape)
        input_data = tf.pad(
            tf.expand_dims(input_data, 1),
            padding,
            #(int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='REFLECT')
        input_data = tf.squeeze(input_data, 1)

        # forward_basis
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = tf.convert_to_tensor(fourier_basis[:, None, :])

        assert(self.filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, self.filter_length)
        fft_window = tf.convert_to_tensor(fft_window, dtype=tf.float32)

        # window the bases
        forward_basis *= fft_window

        # stft transform
        forward_transform = tf.nn.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = tf.math.sqrt(real_part**2 + imag_part**2)
        phase = tf.math.atan2(imag_part, real_part)

        return magnitude, phase


@gin.register
def compute_waveglow_logmel(audio,
                   sample_rate=16000,
                   lo_hz=80.0,
                   hi_hz=7600.0,
                   bins=64,
                   fft_size=2048,
                   overlap=0.75,
                   pad_end=True,
                   mel_samples=1720,
                   filter_length=1024,
                   hop_length=256,
                   win_length=1024):
    print("Computing logmel waveglow style...")
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = tf.expand_dims(audio_norm, 0)
    stft_fn = TacotronSTFT(filter_length=filter_length, hop_length=hop_length, win_length=win_length, sampling_rate=sample_rate, mel_fmin=lo_hz, mel_fmax=hi_hz)
    melspec = stft_fn.mel_spectrogram(audio_norm)
    melspec = tf.squeeze(melspec, 0)
    if mel_samples is not None:
        melspec = melspec[:, :mel_samples]
    return melspec


@gin.register
def compute_logmel(audio,
                   sample_rate=16000,
                   lo_hz=80.0,
                   hi_hz=7600.0,
                   bins=64,
                   fft_size=2048,
                   overlap=0.75,
                   pad_end=True,
                   mel_samples=None):
  print(f"sample rate: {sample_rate}, bin: {bins}, fft_size: {fft_size}, overlap: {overlap}")
  mel = compute_mel(audio, sample_rate, lo_hz, hi_hz, bins, fft_size, overlap, pad_end)
  print("mel shape in compute_logmel: ", mel.shape)
  # Make sure it matches the mel_sample shape
  if mel_samples is not None:
      mel = mel[:, :mel_samples, :]
  print("final mel shape in compute_logmel: ", mel.shape)
  return safe_log(mel)


@gin.register
def compute_mel_spec_process(mel_spec,
                sample_rate=16000,
                lo_hz=0.0,
                hi_hz=8000.0,
                bins=64,
                fft_size=2048,
                overlap=0.75,
                pad_end=True):
  """Processes a given Mel Spectrogram."""
  mag = compute_mag_mel_spec(mel_spec, fft_size, overlap, pad_end)
  print("mel spec mag: ", mag)
  #mag = tf_float32(mel_spec)
  # num_spectrogram_bins = int(mag.shape[-1])
  num_spectrogram_bins = tf.cast(tf.shape(mag)[-1], tf.int32)
  if not bins:
    bins = int(fft_size / 4) + 1
  linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
      bins, num_spectrogram_bins, sample_rate, lo_hz, hi_hz)
#   print("linear to mel matrix: ", linear_to_mel_matrix)
  mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
  mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
  print("Final shape of mel spec: ", mel.shape)
  return mel

@gin.register
def compute_mfcc(audio,
                 sample_rate=16000,
                 lo_hz=20.0,
                 hi_hz=8000.0,
                 fft_size=1024,
                 mel_bins=128,
                 mfcc_bins=13,
                 overlap=0.75,
                 pad_end=True):
  """Calculate Mel-frequency Cepstral Coefficients."""
  logmel = compute_logmel(
      audio,
      sample_rate=sample_rate,
      lo_hz=lo_hz,
      hi_hz=hi_hz,
      bins=mel_bins,
      fft_size=fft_size,
      overlap=overlap,
      pad_end=pad_end)
  #print("logmel: ", logmel)
  print("Original MFCC logmel shape: ", logmel.shape) # Shape is (1, 1723, 128) --> (1, time dim, mel bins)
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(logmel)
  print("Original MFCC shape: ", mfccs.shape)
  return mfccs[..., :mfcc_bins]

@gin.register
def compute_mfcc_mel_spec(mel_spec,
                 sample_rate=16000,
                 lo_hz=20.0,
                 hi_hz=8000.0,
                 fft_size=1024,
                 mel_bins=80, # Default was 128
                 mfcc_bins=13,
                 overlap=0.75,
                 pad_end=True):
  """Calculate Mel-frequency Cepstral Coefficients."""
  # The log mel spectrogram needs to be of shape (1, time, mel bins)
  # Need to take absolute value to deal with the negatives from the waveglow mel specs
  #logmel = safe_log(tf.abs(mel_spec)) # This will set any value <= 1e-5 to be 1e-5. So negative nums don't work!
  # Note that waveglow mel specs ARE log mel spectrograms! So don't need to take additional log
  print("passing in the log mel spectrogram as is...")
  print("Need to transpose the direct output spectrogram from regnet as it's of the shape: (1, mel bins, time) --> (1, 80, 1720)")
  
  # If get regnet output directly
  if len(mel_spec.shape) < 3:
      mel_spec = tf.expand_dims(tf.convert_to_tensor(mel_spec), axis=0)

  if mel_spec.shape[-1] != mel_bins:
      mel_spec = tf.transpose(mel_spec, perm=[0,2,1])

  logmel = mel_spec

  print("Mel spec MFCC logmel shape: ", logmel.shape)
  print("log mel values: ", logmel)
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(logmel)
  print("mel spec mfcc shape: ", mfccs.shape)
  return mfccs[..., :mfcc_bins] # equivalent to mfccs[:,:,:mfcc_bins]



def diff(x, axis=-1):
  """Take the finite difference of a tensor along an axis.

  Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.

  Returns:
    d: Tensor with size less than x by 1 along the difference dimension.

  Raises:
    ValueError: Axis out of range for tensor.
  """
  shape = x.shape.as_list()
  if axis >= len(shape):
    raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                     (axis, len(shape)))

  begin_back = [0 for _ in range(len(shape))]
  begin_front = [0 for _ in range(len(shape))]
  begin_front[axis] = 1

  shape[axis] -= 1
  slice_front = tf.slice(x, begin_front, shape)
  slice_back = tf.slice(x, begin_back, shape)
  d = slice_front - slice_back
  return d


def amplitude_to_db(amplitude, use_tf=False):
  """Converts amplitude to decibels."""
  lib = tf if use_tf else np
  log10 = (lambda x: tf.math.log(x) / tf.math.log(10.0)) if use_tf else np.log10
  amin = 1e-20  # Avoid log(0) instabilities.
  db = log10(lib.maximum(amin, amplitude))
  db *= 20.0
  return db


def db_to_amplitude(db):
  """Converts decibels to amplitude."""
  return 10.0**(db / 20.0)


@gin.register
def compute_loudness(audio,
                     sample_rate=16000,
                     frame_rate=250,
                     n_fft=2048,
                     range_db=LD_RANGE,
                     ref_db=20.7,
                     use_tf=False):
  """Perceptual loudness in dB, relative to white noise, amplitude=1.

  Function is differentiable if use_tf=True.
  Args:
    audio: Numpy ndarray or tensor. Shape [batch_size, audio_length] or
      [batch_size,].
    sample_rate: Audio sample rate in Hz.
    frame_rate: Rate of loudness frames in Hz.
    n_fft: Fft window size.
    range_db: Sets the dynamic range of loudness in decibles. The minimum
      loudness (per a frequency bin) corresponds to -range_db.
    ref_db: Sets the reference maximum perceptual loudness as given by
      (A_weighting + 10 * log10(abs(stft(audio))**2.0). The default value
      corresponds to white noise with amplitude=1.0 and n_fft=2048. There is a
      slight dependence on fft_size due to different granularity of perceptual
      weighting.
    use_tf: Make function differentiable by using tensorflow.

  Returns:
    Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
  """
  if sample_rate % frame_rate != 0:
    raise ValueError(
        'frame_rate: {} must evenly divide sample_rate: {}.'
        'For default frame_rate: 250Hz, suggested sample_rate: 16kHz or 48kHz'
        .format(frame_rate, sample_rate))

  # Pick tensorflow or numpy.
  lib = tf if use_tf else np

  # Make inputs tensors for tensorflow.
  audio = tf_float32(audio) if use_tf else audio

  # Temporarily a batch dimension for single examples.
  is_1d = (len(audio.shape) == 1)
  audio = audio[lib.newaxis, :] if is_1d else audio

  # Take STFT.
  hop_size = sample_rate // frame_rate
  overlap = 1 - hop_size / n_fft
  stft_fn = stft if use_tf else stft_np
  s = stft_fn(audio, frame_size=n_fft, overlap=overlap, pad_end=True)
  print("ddsp author hop size: ", hop_size)
  print("overlap: ", overlap)
  print("Computing loudness after stft shape: ", s.shape)

  # Compute power.
  amplitude = lib.abs(s)
  power_db = amplitude_to_db(amplitude, use_tf=use_tf)
  print("power_db shape: ", power_db.shape)

  # Perceptual weighting.
  frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
  a_weighting = librosa.A_weighting(frequencies)[lib.newaxis, lib.newaxis, :]
  loudness = power_db + a_weighting
  print("loudness shape: ", loudness.shape)

  # Set dynamic range.
  loudness -= ref_db
  loudness = lib.maximum(loudness, -range_db)
  mean = tf.reduce_mean if use_tf else np.mean

  # Average over frequency bins.
  loudness = mean(loudness, axis=-1)

  # Remove temporary batch dimension.
  loudness = loudness[0] if is_1d else loudness
  print("shape of audio: ", audio.shape)
  print("sample rate: ", sample_rate)

  # Compute expected length of loudness vector
  n_secs = audio.shape[-1] / float(
      sample_rate)  # `n_secs` can have milliseconds
  print("n_secs: ", n_secs)
  expected_len = int(n_secs * frame_rate)
  print("expected length: ", expected_len)
  print("shape of loudness before pad/trim: ", loudness.shape)

  # Pad with `-range_db` noise floor or trim vector
  loudness = pad_or_trim_to_expected_length(
      loudness, expected_len, -range_db, use_tf=use_tf)
  return loudness


@gin.register
def compute_loudness_mel_spec(mel_spec, audio,
                     sample_rate=44100,
                     frame_rate=172,
                     n_fft=1024,
                     range_db=LD_RANGE,
                     ref_db=20.7,
                     use_tf=False):
  """Perceptual loudness in dB, relative to white noise, amplitude=1.

  Function is differentiable if use_tf=True.
  Args:
    mel_spec: Mel spectrogram of the audio
    sample_rate: Audio sample rate in Hz.
    frame_rate: Rate of loudness frames in Hz.
    n_fft: Fft window size.
    range_db: Sets the dynamic range of loudness in decibles. The minimum
      loudness (per a frequency bin) corresponds to -range_db.
    ref_db: Sets the reference maximum perceptual loudness as given by
      (A_weighting + 10 * log10(abs(stft(audio))**2.0). The default value
      corresponds to white noise with amplitude=1.0 and n_fft=2048. There is a
      slight dependence on fft_size due to different granularity of perceptual
      weighting.
    use_tf: Make function differentiable by using tensorflow.

  Returns:
    Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
  """
#   if sample_rate % frame_rate != 0:
#     raise ValueError(
#         'frame_rate: {} must evenly divide sample_rate: {}.'
#         'For default frame_rate: 250Hz, suggested sample_rate: 16kHz or 48kHz'
#         .format(frame_rate, sample_rate))
    
  print("s shape input to compute_loudness function: ", audio.shape)

  # Pick tensorflow or numpy.
  lib = tf if use_tf else np

  # Make inputs tensors for tensorflow.
  # if audio sampling rate=44100, dims will be (80, 1720)
  # if audio sampling rate=22050, dims will be (80, 860)
  mel_spec = tf_float32(mel_spec) if use_tf else mel_spec
    
  # Make inputs tensors for tensorflow.
  audio = tf_float32(audio) if use_tf else audio

  # Temporarily a batch dimension for single examples.
  is_1d = (len(audio.shape) == 1)
  audio = audio[lib.newaxis, :] if is_1d else audio

  # Temporarily a batch dimension for single examples.
  is_1d = (len(mel_spec.shape) == 1)
  mel_spec = mel_spec[lib.newaxis, :] if is_1d else mel_spec

  # Take STFT.
  print("s shape before stft shape: ", audio.shape)
  hop_size = sample_rate // frame_rate #256
  overlap = 1 - hop_size / n_fft
  stft_fn = stft if use_tf else stft_np
  s = stft_fn(audio, frame_size=n_fft, overlap=overlap, pad_end=True)
  print("hop size: ", hop_size)
  print("overlap: ", overlap)
  print("Computing s after stft shape: ", s.shape)
  
  is_2d = (len(mel_spec.shape) == 2)
  mel_spec = mel_spec[lib.newaxis, :] if is_2d else mel_spec
  print("mel spec shape before computing power: ", mel_spec.shape)

  # Compute power.
  amplitude = lib.abs(mel_spec)
  power_db = amplitude_to_db(amplitude, use_tf=use_tf)
  print("power_db: ", power_db)
    
  # Computing power with directly from mel spec
  power_db_lib = librosa.power_to_db(mel_spec, ref=10)
  print("power_db_librosa: ", power_db_lib)
    
  # Perceptual weighting.
#   if sample_rate == 44100:
#       nfft_match = 3438
#   elif sample_rate == 22050:
#       nfft_match = 1718
#   else:
#       nfft_match = n_fft
  # Num mel channels is 80 by default (currently at least)
  nfft_match = 158
  frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=nfft_match) # returns (1 + n_ftt/2,) dims
  a_weighting = librosa.A_weighting(frequencies)[lib.newaxis, lib.newaxis, :]
  # For broadcasting to match 2D mel spec, weights also need to be 2D, not 3D
  #a_weighting = librosa.A_weighting(frequencies)[lib.newaxis, :]
  print(f"shape of power_db: {power_db.shape} and shape of weights: {a_weighting.shape}")
  loudness = power_db + a_weighting

  # Set dynamic range.
  loudness -= ref_db
  loudness = lib.maximum(loudness, -range_db)
  mean = tf.reduce_mean if use_tf else np.mean

  # Average over frequency bins.
  loudness = mean(loudness, axis=-1)

  # Remove temporary batch dimension.
  loudness = loudness[0] if is_1d else loudness

  # Compute expected length of loudness vector (using original audio as reference, NOT mel spec)
  n_secs = mel_spec.shape[-1] / float(
      sample_rate)  # `n_secs` can have milliseconds
  #expected_len = int(n_secs * frame_rate) # To match num_mel_channels=80 for the mel specs
  expected_len = loudness.shape[-1]
  print("expected length: ", expected_len)
  print("shape of loudness before pad/trim: ", loudness.shape)

  # Pad with `-range_db` noise floor or trim vector
  loudness = pad_or_trim_to_expected_length(
      loudness, expected_len, -range_db, use_tf=use_tf)
  return loudness, s


@gin.register
def compute_f0(audio, sample_rate, frame_rate, viterbi=True):
  """Fundamental frequency (f0) estimate using CREPE.

  This function is non-differentiable and takes input as a numpy array.
  Args:
    audio: Numpy ndarray of single audio example. Shape [audio_length,].
    sample_rate: Sample rate in Hz.
    frame_rate: Rate of f0 frames in Hz.
    viterbi: Use Viterbi decoding to estimate f0.

  Returns:
    f0_hz: Fundamental frequency in Hz. Shape [n_frames,].
    f0_confidence: Confidence in Hz estimate (scaled [0, 1]). Shape [n_frames,].
  """

  n_secs = len(audio) / float(sample_rate)  # `n_secs` can have milliseconds
  crepe_step_size = 1000 / frame_rate  # milliseconds
  expected_len = int(n_secs * frame_rate)
  audio = np.asarray(audio)

  # Compute f0 with crepe.
  _, f0_hz, f0_confidence, _ = crepe.predict(
      audio,
      sr=sample_rate,
      viterbi=viterbi,
      step_size=crepe_step_size,
      center=False,
      verbose=0)

  # Postprocessing on f0_hz
  f0_hz = pad_or_trim_to_expected_length(f0_hz, expected_len, 0)  # pad with 0
  f0_hz = f0_hz.astype(np.float32)

  # Postprocessing on f0_confidence
  f0_confidence = pad_or_trim_to_expected_length(f0_confidence, expected_len, 1)
  f0_confidence = np.nan_to_num(f0_confidence)   # Set nans to 0 in confidence
  f0_confidence = f0_confidence.astype(np.float32)
  return f0_hz, f0_confidence


def compute_rms_energy(audio,
                       sample_rate=16000,
                       frame_rate=250,
                       frame_size=2048):
  """Compute root mean squared energy of audio."""
  n_secs = len(audio) / float(sample_rate)  # `n_secs` can have milliseconds
  expected_len = int(n_secs * frame_rate)

  audio = tf_float32(audio)

  hop_size = sample_rate // frame_rate
  audio_frames = tf.signal.frame(audio, frame_size, hop_size, pad_end=True)
  rms_energy = tf.reduce_mean(audio_frames**2.0, axis=-1)**0.5
  return pad_or_trim_to_expected_length(rms_energy, expected_len, use_tf=True)


def compute_power(audio,
                  sample_rate=16000,
                  frame_rate=250,
                  frame_size=1024,
                  range_db=LD_RANGE,
                  ref_db=20.7):
  """Compute power of audio in dB."""
  # TODO(hanoih@): enable `use_tf` to be True or False like `compute_loudness`
  rms_energy = compute_rms_energy(audio, sample_rate, frame_rate, frame_size)
  power_db = amplitude_to_db(rms_energy**2, use_tf=True)
  # Set dynamic range.
  power_db -= ref_db
  power_db = tf.maximum(power_db, -range_db)
  return power_db


def pad_or_trim_to_expected_length(vector,
                                   expected_len,
                                   pad_value=0,
                                   len_tolerance=20,
                                   use_tf=False):
  """Make vector equal to the expected length.

  Feature extraction functions like `compute_loudness()` or `compute_f0` produce
  feature vectors that vary in length depending on factors such as `sample_rate`
  or `hop_size`. This function corrects vectors to the expected length, warning
  the user if the difference between the vector and expected length was
  unusually high to begin with.

  Args:
    vector: Numpy 1D ndarray. Shape [vector_length,]
    expected_len: Expected length of vector.
    pad_value: Value to pad at end of vector.
    len_tolerance: Tolerance of difference between original and desired vector
      length.
    use_tf: Make function differentiable by using tensorflow.

  Returns:
    vector: Vector with corrected length.

  Raises:
    ValueError: if `len(vector)` is different from `expected_len` beyond
    `len_tolerance` to begin with.
  """
  expected_len = int(expected_len)
  vector_len = int(vector.shape[-1])

  if abs(vector_len - expected_len) > len_tolerance:
    # Ensure vector was close to expected length to begin with
    raise ValueError('Vector length: {} differs from expected length: {} '
                     'beyond tolerance of : {}'.format(vector_len,
                                                       expected_len,
                                                       len_tolerance))
  # Pick tensorflow or numpy.
  lib = tf if use_tf else np

  is_1d = (len(vector.shape) == 1)
  vector = vector[lib.newaxis, :] if is_1d else vector

  # Pad missing samples
  if vector_len < expected_len:
    n_padding = expected_len - vector_len
    vector = lib.pad(
        vector, ((0, 0), (0, n_padding)),
        mode='constant',
        constant_values=pad_value)
  # Trim samples
  elif vector_len > expected_len:
    vector = vector[..., :expected_len]

  # Remove temporary batch dimension.
  vector = vector[0] if is_1d else vector
  return vector


def reset_crepe():
  """Reset the global state of CREPE to force model re-building."""
  for k in crepe.core.models:
    crepe.core.models[k] = None
