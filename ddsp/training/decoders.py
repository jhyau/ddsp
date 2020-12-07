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
"""Library of decoder layers."""

import ddsp
from ddsp.training import nn
import gin
import tensorflow as tf

tfkl = tf.keras.layers


# ------------------ Decoders --------------------------------------------------
@gin.register
class RnnFcDecoder(nn.OutputSplitsLayer):
  """RNN and FC stacks for f0 and loudness."""

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               ch=512,
               layers_per_stack=3,
               input_keys=('ld_scaled', 'f0_scaled', 'z'),
               output_splits=(('amps', 1), ('harmonic_distribution', 40)),
               **kwargs):
    super().__init__(
        input_keys=input_keys, output_splits=output_splits, **kwargs)
    stack = lambda: nn.FcStack(ch, layers_per_stack)

    # Layers.
    self.input_stacks = [stack() for k in self.input_keys]
    self.rnn = nn.Rnn(rnn_channels, rnn_type)
    self.out_stack = stack()

  def compute_output(self, *inputs):
    # Initial processing.
    inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]

    # Run an RNN over the latents.
    x = tf.concat(inputs, axis=-1)
    x = self.rnn(x)
    x = tf.concat(inputs + [x], axis=-1)

    # Final processing.
    return self.out_stack(x)

@gin.register
class TemporalCNNFcDecoder(RnnFcDecoder):
  """Temporal and FC stacks for f0 and loudness."""

  def __init__(self,
               temporal_cnn_channels=512,
               window_size=30,
               ch=512,
               layers_per_stack=3,
               input_keys=('ld_scaled', 'f0_scaled', 'z'),
               output_splits=(('amps', 1), ('harmonic_distribution', 40)),
               name=None):
    super().__init__(rnn_channels=1, ch=ch, layers_per_stack=layers_per_stack, input_keys=input_keys, output_splits=output_splits, name=name)
    self.rnn = nn.temporal_cnn(temporal_cnn_channels, window_size)

@gin.register
class FcDecoder(nn.OutputSplitsLayer):
  """Fully connected network decoder."""

  def __init__(self,
               fc_units=512,
               hidden_layers=3,
               input_keys=('object_embedding'),
               output_splits=(('frequencies', 200), ('gains', 200), ('dampings', 200)),
               **kwargs):
    super().__init__(
        input_keys=input_keys, output_splits=output_splits, **kwargs)
    stack = lambda: nn.FcStack(fc_units, hidden_layers)
    self.input_stacks = [stack() for k in self.input_keys]
    self.out_stack = stack()

  def compute_output(self, *inputs):
    # Initial processing.
    inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]
    x = tf.concat(inputs, axis=-1)

    # Final processing.
    return self.out_stack(x)


@gin.register
class MultiDecoder(tfkl.Layer):
  """Combine multiple decoders into one."""

  def __init__(self,
               decoder_list,
               **kwargs):
    super().__init__(**kwargs)
    self.decoder_list = decoder_list

  def call(self, inputs):
    """Updates conditioning with dictionary of decoder outputs."""
    conditioning = ddsp.core.copy_if_tf_function(inputs)
    for dec in self.decoder_list:
      x = dec(conditioning)
      if isinstance(x, dict):
        conditioning.update(x)
      else:
        raise ValueError('Encoder must output a dictionary of signals.')
    return conditioning