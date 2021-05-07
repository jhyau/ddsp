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
"""Library of synthesizer functions."""
import math

from ddsp import core
from ddsp import processors
import gin
import tensorflow.compat.v2 as tf


@gin.register
class TensorToAudio(processors.Processor):
  """Identity "synth" returning input samples with channel dimension removed."""

  def __init__(self, name='tensor_to_audio'):
    super().__init__(name=name)

  def get_controls(self, samples):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      samples: 3-D Tensor of "controls" (really just samples), of shape
        [batch, time, 1].

    Returns:
      Dictionary of tensors of synthesizer controls.
    """
    return {'samples': samples}

  def get_signal(self, samples):
    """"Synthesize" audio by removing channel dimension from input samples.

    Args:
      samples: 3-D Tensor of "controls" (really just samples), of shape
        [batch, time, 1].

    Returns:
      A tensor of audio with shape [batch, time].
    """
    return tf.squeeze(samples, 2)


@gin.register
class Harmonic(processors.Processor):
  """Synthesize audio with a bank of harmonic sinusoidal oscillators."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               scale_fn=core.exp_sigmoid,
               normalize_below_nyquist=True,
               name='harmonic'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.scale_fn = scale_fn
    self.normalize_below_nyquist = normalize_below_nyquist

  def get_controls(self,
                   amplitudes,
                   harmonic_distribution,
                   f0_hz):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      amplitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      harmonic_distribution: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_harmonics].
      f0_hz: Fundamental frequencies in hertz. Shape [batch, time, 1].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the amplitudes.
    if self.scale_fn is not None:
      amplitudes = self.scale_fn(amplitudes)
      harmonic_distribution = self.scale_fn(harmonic_distribution)

    # Bandlimit the harmonic distribution.
    if self.normalize_below_nyquist:
      n_harmonics = int(harmonic_distribution.shape[-1])
      harmonic_frequencies = core.get_harmonic_frequencies(f0_hz,
                                                           n_harmonics)
      harmonic_distribution = core.remove_above_nyquist(harmonic_frequencies,
                                                        harmonic_distribution,
                                                        self.sample_rate)

    # Normalize
    harmonic_distribution /= tf.reduce_sum(harmonic_distribution,
                                           axis=-1,
                                           keepdims=True)

    return {'amplitudes': amplitudes,
            'harmonic_distribution': harmonic_distribution,
            'f0_hz': f0_hz}

  def get_signal(self, amplitudes, harmonic_distribution, f0_hz):
    """Synthesize audio with harmonic synthesizer from controls.

    Args:
      amplitudes: Amplitude tensor of shape [batch, n_frames, 1]. Expects
        float32 that is strictly positive.
      harmonic_distribution: Tensor of shape [batch, n_frames, n_harmonics].
        Expects float32 that is strictly positive and normalized in the last
        dimension.
      f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
        n_frames, 1].

    Returns:
      signal: A tensor of harmonic waves of shape [batch, n_samples].
    """
    signal = core.harmonic_synthesis(
        frequencies=f0_hz,
        amplitudes=amplitudes,
        harmonic_distribution=harmonic_distribution,
        n_samples=self.n_samples,
        sample_rate=self.sample_rate)
    return signal


@gin.register
class FilteredNoise(processors.Processor):
  """Synthesize audio by filtering white noise."""

  def __init__(self,
               n_samples=64000,
               window_size=257,
               scale_fn=core.exp_sigmoid,
               initial_bias=-5.0,
               name='filtered_noise'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.window_size = window_size
    self.scale_fn = scale_fn
    self.initial_bias = initial_bias

  def get_controls(self, magnitudes):
    """Convert network outputs into a dictionary of synthesizer controls.

    Args:
      magnitudes: 3-D Tensor of synthesizer parameters, of shape [batch, time,
        n_filter_banks].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the magnitudes.
    if self.scale_fn is not None:
      magnitudes = self.scale_fn(magnitudes + self.initial_bias)

    return {'magnitudes': magnitudes}

  def get_signal(self, magnitudes):
    """Synthesize audio with filtered white noise.

    Args:
      magnitudes: Magnitudes tensor of shape [batch, n_frames, n_filter_banks].
        Expects float32 that is strictly positive.

    Returns:
      signal: A tensor of harmonic waves of shape [batch, n_samples, 1].
    """
    batch_size = int(magnitudes.shape[0])
    signal = tf.random.uniform(
        [batch_size, self.n_samples], minval=-1.0, maxval=1.0)
    return core.frequency_filter(signal,
                                 magnitudes,
                                 window_size=self.window_size)


@gin.register
class Wavetable(processors.Processor):
  """Synthesize audio from a series of wavetables."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               scale_fn=core.exp_sigmoid,
               name='wavetable'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.scale_fn = scale_fn

  def get_controls(self,
                   amplitudes,
                   wavetables,
                   f0_hz):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      amplitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      wavetables: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_harmonics].
      f0_hz: Fundamental frequencies in hertz. Shape [batch, time, 1].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the amplitudes.
    if self.scale_fn is not None:
      amplitudes = self.scale_fn(amplitudes)
      wavetables = self.scale_fn(wavetables)

    return  {'amplitudes': amplitudes,
             'wavetables': wavetables,
             'f0_hz': f0_hz}

  def get_signal(self, amplitudes, wavetables, f0_hz):
    """Synthesize audio with wavetable synthesizer from controls.

    Args:
      amplitudes: Amplitude tensor of shape [batch, n_frames, 1]. Expects
        float32 that is strictly positive.
      wavetables: Tensor of shape [batch, n_frames, n_wavetable].
      f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
        n_frames, 1].

    Returns:
      signal: A tensor of of shape [batch, n_samples].
    """
    wavetables = core.resample(wavetables, self.n_samples)
    signal = core.wavetable_synthesis(amplitudes=amplitudes,
                                      wavetables=wavetables,
                                      frequencies=f0_hz,
                                      n_samples=self.n_samples,
                                      sample_rate=self.sample_rate)
    return signal


@gin.register
class Sinusoidal(processors.Processor):
  """Synthesize audio with a bank of arbitrary sinusoidal oscillators."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               amp_scale_fn=core.exp_sigmoid,
               amp_resample_method='window',
               freq_scale_fn=core.frequencies_sigmoid,
               freq_scale='bark',
               hz_max=8000,
               name='sinusoidal'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.amp_scale_fn = amp_scale_fn
    self.amp_resample_method = amp_resample_method
    self.freq_scale_fn = freq_scale_fn
    self.freq_scale = freq_scale
    self.hz_max = hz_max

  def get_controls(self, amplitudes, frequencies):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      amplitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_sinusoids].
      frequencies: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_sinusoids]. Expects strictly positive in Hertz.

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the inputs.
    if self.amp_scale_fn is not None:
      amplitudes = self.amp_scale_fn(amplitudes)

    if self.freq_scale_fn is not None:
      frequencies = self.freq_scale_fn(frequencies, scale=self.freq_scale, hz_max=self.hz_max)
      amplitudes = core.remove_above_nyquist(frequencies,
                                             amplitudes,
                                             self.sample_rate)

    return {'amplitudes': amplitudes,
            'frequencies': frequencies}

  def get_signal(self, amplitudes, frequencies):
    """Synthesize audio with sinusoidal synthesizer from controls.

    Args:
      amplitudes: Amplitude tensor of shape [batch, n_frames, n_sinusoids].
        Expects float32 that is strictly positive.
      frequencies: Tensor of shape [batch, n_frames, n_sinusoids].
        Expects float32 in Hertz that is strictly positive.

    Returns:
      signal: A tensor of harmonic waves of shape [batch, n_samples].
    """
    # Create sample-wise envelopes.
    amplitude_envelopes = core.resample(amplitudes, self.n_samples,
                                        method=self.amp_resample_method)
    frequency_envelopes = core.resample(frequencies, self.n_samples)

    signal = core.oscillator_bank(frequency_envelopes=frequency_envelopes,
                                  amplitude_envelopes=amplitude_envelopes,
                                  sample_rate=self.sample_rate)
    return signal


@gin.register
class Impact(processors.Processor):
  """Synthesize an impact profile with a Gaussian force contact model."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               mag_scale_fn=core.exp_sigmoid,
               resample_method='window',
               max_tau=0.003,
               max_impact_frequency=30,
               initial_bias=-1.5,
               timing_adjust=True,
               include_noise=True,
               name='impact'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.mag_scale_fn = mag_scale_fn
    self.resample_method = resample_method
    self.max_tau = max_tau
    self.max_impact_frequency=max_impact_frequency
    self.initial_bias = initial_bias
    self.timing_adjust = timing_adjust
    self.include_noise = include_noise

  def get_controls(self, magnitudes, stdevs, taus, tau_bias):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      magnitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      stdevs: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      taus: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      tau_bias: 3-D Tensor of synthesizer controls, of shape
        [batch, 1, 1].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the inputs.
    if self.mag_scale_fn is not None:
      if self.include_noise:
        noise = tf.abs(stdevs) * tf.random.normal(stdevs.shape, dtype=tf.float32)
        magnitudes = self.mag_scale_fn(magnitudes + noise + self.initial_bias)
      else:
        magnitudes = self.mag_scale_fn(magnitudes + self.initial_bias)
      # taus = self.max_tau * tf.nn.sigmoid(2.0 * tau_bias) * tf.nn.sigmoid(taus) + 1.0/self.sample_rate
      # noise = tau_bias + tau_bias * tf.random.normal(taus.shape, dtype=tf.float32)
      taus = core.exp_sigmoid(tau_bias + taus, exponent=4.0, max_value=self.max_tau, threshold=3.0/self.sample_rate)

    return {'magnitudes': magnitudes,
            'taus': taus}


  def hertz_gaussian(self, peak_times, tau):
    t = tf.reshape(tf.range(self.n_samples, dtype = tf.float32) / self.sample_rate, (1, -1, 1))
    # impulses =  tf.exp(-6/tf.square(tau) * tf.square(t - peak_times - tau / 2))
    impulses =  tf.exp(-6/tf.square(tau) * tf.square(t - peak_times))
    return impulses

  def hertz_sine(self, peak_times, tau):
    t = tf.reshape(tf.range(self.n_samples, dtype = tf.float32) / self.sample_rate, (1, -1, 1))
    impulses =  tf.sin(math.pi*(t - peak_times) / tau)
    impulses = impulses * tf.cast(tf.logical_and(t >= peak_times, t <= (peak_times+tau)), tf.float32)
    return impulses

  def get_signal(self, magnitudes, taus):
    """Synthesize audio with sinusoidal synthesizer from controls.

    Args:
      magnitudes: magnitude tensor of shape [batch, n_frames, 1].
        Expects float32 that is strictly positive.
      stdevs: Tensor of shape [batch, n_frames, 1].
        Expects float32 in that is strictly positive.
      taus: Tensor of shape [batch, n_frames, 1].
        Expects float32 in that is strictly positive.

    Returns:
      signal: A tensor of the force impulse profile of shape [batch, n_samples].
    """
    # Create sample-wise envelopes.
    weight_distance = 100
    diff_order = 1
    # magnitude_diffs = tf.experimental.numpy.diff(magnitudes, n=diff_order, axis=1)
    # magnitude_diffs = tf.concat((tf.zeros((tf.shape(magnitude_diffs)[0], int(math.floor(diff_order/2)), 1), dtype=tf.float32),
    #                             magnitude_diffs,
    #                             tf.zeros((tf.shape(magnitude_diffs)[0], int(math.ceil(diff_order/2)), 1), dtype=tf.float32)), axis=1)
    # magnitude_envelopes = core.resample(((-1)**diff_order) *  magnitude_diffs, self.n_samples,
    magnitude_envelopes = core.resample(magnitudes, self.n_samples,
                                        method=self.resample_method)
    taus = core.resample(taus, self.n_samples,
                          method=self.resample_method)

    window_size = int(self.sample_rate / self.max_impact_frequency)
    magnitude_envelopes = tf.expand_dims(magnitude_envelopes, axis=1)
    vals, inds = tf.nn.max_pool_with_argmax(magnitude_envelopes, window_size, window_size, 'SAME')
    # Use a weighted average of magnitude to select peak time so that things can shift around
    if self.timing_adjust:
      augmented_inds = tf.concat([inds - weight_distance, inds, inds + weight_distance], axis=-1)
      b,w,h,c = magnitude_envelopes.get_shape().as_list()
      mags_pooled = tf.gather(tf.reshape(magnitude_envelopes, shape=[b*w*h*c]), augmented_inds)
      weighted_inds = tf.reduce_sum(tf.cast(augmented_inds, dtype=tf.float32) * mags_pooled, axis=-1) / tf.reduce_sum(mags_pooled, axis=-1)
      peak_times = tf.cast(weighted_inds / self.sample_rate, dtype=tf.float32)
    else:
      peak_times = tf.squeeze(tf.cast(inds / self.sample_rate, dtype=tf.float32), axis=3)
    
    scale_heights = tf.squeeze(vals, axis=3)
    taus = tf.expand_dims(taus, axis=1)
    b,w,h,c = taus.get_shape().as_list()
    taus_pooled = tf.gather(tf.reshape(taus,shape= [b*w*h*c,]),inds)
    taus_pooled = tf.squeeze(taus_pooled, axis=3)
    basis_impulses = self.hertz_gaussian(peak_times, taus_pooled)
    signal = tf.reduce_sum(scale_heights * basis_impulses, axis=2)
    return signal

@gin.register
class ModalFIR(processors.Processor):
  """Synthesize a modal FIR with exponentially decaying sinusoidal oscillators."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               amp_scale_fn=core.exp_sigmoid,
               amp_resample_method='window',
               freq_scale_fn=core.frequencies_critical_bands,
               hz_max=8000.0,
               freq_scale='bark',
               initial_bias=-1.5,
               name='modal_fir'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.amp_scale_fn = amp_scale_fn
    self.amp_resample_method = amp_resample_method
    self.freq_scale_fn = freq_scale_fn
    self.hz_max = hz_max
    self.freq_scale = freq_scale
    self.initial_bias = initial_bias

  def get_controls(self, gains, frequencies, dampings):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      gains: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_sinusoids].
      frequencies: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_sinusoids].
      dampings: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_sinusoids].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the inputs.
    if self.amp_scale_fn is not None:
      # gains = 0.001 * self.amp_scale_fn(gains + self.initial_bias, exponent=3.0)
      gains = tf.nn.softmax(gains)
      # dampings = 10.0 / self.amp_scale_fn(4.0 * dampings + self.initial_bias)
      # dampings = 0.05 * self.freq_scale_fn(dampings, hz_min=0.0, hz_max=100000.0)
      dampings = 10000 * self.amp_scale_fn(dampings + self.initial_bias, exponent=4.0)

    if self.freq_scale_fn is not None:
      if len(frequencies.shape) == 2:
        frequencies = tf.expand_dims(frequencies, axis=1)
      frequencies = self.freq_scale_fn(frequencies, hz_min=10.0, hz_max=self.hz_max, scale=self.freq_scale)
    gains = core.remove_above_nyquist(frequencies,
                                            gains,
                                            self.sample_rate)
    return {'gains': gains,
            'frequencies': frequencies,
            'dampings': dampings}

  def get_signal(self, gains, frequencies, dampings):
    """Synthesize audio with sinusoidal synthesizer from controls.

    Args:
      gains: Gains tensor of shape [batch, n_frames, n_sinusoids].
        Expects float32 that is strictly positive.
      frequencies: Tensor of shape [batch, n_frames, n_sinusoids].
        Expects float32 in Hertz that is strictly positive.
      dampings: Tensor of shape [batch, n_frames, n_sinusoids].
        Expects float32 in Hertz that is strictly positive.

    Returns:
      signal: A tensor of exponentially decaying modal frequencies of shape [batch, n_samples].
    """
    # Create sample-wise envelopes.
    t = tf.expand_dims(tf.cast(tf.range(self.n_samples)/self.sample_rate, dtype=tf.float32), axis=1)
    amplitude_envelopes = gains * tf.exp(-dampings * t)
    frequency_envelopes = frequencies * tf.ones_like(amplitude_envelopes)
    ir_half = core.oscillator_bank(frequency_envelopes=frequency_envelopes,
                                   amplitude_envelopes=amplitude_envelopes,
                                   sample_rate=self.sample_rate)
    signal = tf.concat((tf.zeros_like(ir_half), ir_half), axis=1)
    return signal