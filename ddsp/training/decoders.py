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
class FcDecoder(Decoder):
  """Fully connected network decoder."""

  def __init__(self,
               fc_units=512,
               hidden_layers=3,
               input_keys=('object_embedding'),
               output_splits=(('frequencies', 200), ('gains', 200), ('dampings', 200)),
               activity_regularizer=None,
               name=None):
    super().__init__(output_splits=output_splits, name=name)
    self.out_stack = nn.FcStack(fc_units, hidden_layers)
    self.input_keys = input_keys
    self.dense_out = tfkl.Dense(self.n_out, activity_regularizer=activity_regularizer)

  def decode(self, conditioning):
    # Initial processing.
    inputs = [conditioning[k] for k in self.input_keys]
    # Concatenate.
    x = tf.concat(inputs, axis=-1)
    # Final processing.
    x = self.out_stack(x)
    return self.dense_out(x)


@gin.register
class MultiDecoder(Decoder):
  """Combine multiple decoders into one."""

  def __init__(self,
               decoder_list,
               name=None):
    output_splits = []
    [output_splits.extend(d.output_splits) for d in decoder_list]
    super().__init__(output_splits=output_splits, name=name)
    self.decoder_list = decoder_list

  def call(self, conditioning):
    """Updates conditioning with dictionary of decoder outputs."""
    conditioning = core.copy_if_tf_function(conditioning)

    for d in self.decoder_list:
      x = d.decode(conditioning)
      outputs = nn.split_to_dict(x, d.output_splits)

      if isinstance(outputs, dict):
        conditioning.update(outputs)
      else:
        raise ValueError('Decoder must output a dictionary of signals.')
    return conditioning