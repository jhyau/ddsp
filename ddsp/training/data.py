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
"""Library of functions to help loading data."""
import glob
import os

from absl import logging
import gin
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio


_AUTOTUNE = tf.data.experimental.AUTOTUNE


# ---------- Base Class --------------------------------------------------------
class DataProvider(object):
  """Base class for returning a dataset."""

  def __init__(self, sample_rate, frame_rate):
    """DataProvider constructor.

    Args:
      sample_rate: Sample rate of audio in the dataset.
      frame_rate: Frame rate of features in the dataset.
    """
    self._sample_rate = sample_rate
    self._frame_rate = frame_rate

  @property
  def sample_rate(self):
    """Return dataset sample rate, must be defined in the constructor."""
    return self._sample_rate

  @property
  def frame_rate(self):
    """Return dataset feature frame rate, must be defined in the constructor."""
    return self._frame_rate

  def get_dataset(self, shuffle):
    """A method that returns a tf.data.Dataset."""
    raise NotImplementedError

  def get_batch(self, batch_size, shuffle=True, repeats=-1):
    """Read dataset.

    Args:
      batch_size: Size of batch.
      shuffle: Whether to shuffle the examples.
      repeats: Number of times to repeat dataset. -1 for endless repeats.

    Returns:
      A batched tf.data.Dataset.
    """
    dataset = self.get_dataset(shuffle)
    dataset = dataset.repeat(repeats)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=_AUTOTUNE)
    return dataset


class TfdsProvider(DataProvider):
  """Base class for reading datasets from TensorFlow Datasets (TFDS)."""

  def __init__(self, name, split, data_dir, sample_rate, frame_rate):
    """TfdsProvider constructor.

    Args:
      name: TFDS dataset name (with optional config and version).
      split: Dataset split to use of the TFDS dataset.
      data_dir: The directory to read TFDS datasets from. Defaults to
        "~/tensorflow_datasets".
      sample_rate: Sample rate of audio in the dataset.
      frame_rate: Frame rate of features in the dataset.
    """
    self._name = name
    self._split = split
    self._data_dir = data_dir
    super().__init__(sample_rate, frame_rate)

  def get_dataset(self, shuffle=True):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the input files.

    Returns:
      dataset: A tf.data.Dataset that reads from TFDS.
    """
    return tfds.load(
        self._name,
        data_dir=self._data_dir,
        split=self._split,
        shuffle_files=shuffle,
        download=False)


@gin.register
class NSynthTfds(TfdsProvider):
  """Parses features in the TFDS NSynth dataset.

  If running on Cloud, it is recommended you set `data_dir` to
  'gs://tfds-data/datasets' to avoid unnecessary downloads.
  """

  def __init__(self,
               name='nsynth/gansynth_subset.f0_and_loudness:2.3.0',
               split='train',
               data_dir='gs://tfds-data/datasets',
               sample_rate=16000,
               frame_rate=250):
    """TfdsProvider constructor.

    Args:
      name: TFDS dataset name (with optional config and version).
      split: Dataset split to use of the TFDS dataset.
      data_dir: The directory to read the prepared NSynth dataset from. Defaults
        to the public TFDS GCS bucket.
      sample_rate: Sample rate of audio in the dataset.
      frame_rate: Frame rate of features in the dataset.
    """
    if data_dir == 'gs://tfds-data/datasets':
      logging.warning(
          'Using public TFDS GCS bucket to load NSynth. If not running on '
          'GCP, this will be very slow, and it is recommended you prepare '
          'the dataset locally with TFDS and set the data_dir appropriately.')
    super().__init__(name, split, data_dir, sample_rate, frame_rate)

  def get_dataset(self, shuffle=True):
    """Returns dataset with slight restructuring of feature dictionary."""
    def preprocess_ex(ex):
      return {
          'pitch':
              ex['pitch'],
          'audio':
              ex['audio'],
          'instrument_source':
              ex['instrument']['source'],
          'instrument_family':
              ex['instrument']['family'],
          'instrument':
              ex['instrument']['label'],
          'f0_hz':
              ex['f0']['hz'],
          'f0_confidence':
              ex['f0']['confidence'],
          'loudness_db':
              ex['loudness']['db'],
      }
    dataset = super().get_dataset(shuffle)
    dataset = dataset.map(preprocess_ex, num_parallel_calls=_AUTOTUNE)
    return dataset


class RecordProvider(DataProvider):
  """Class for reading records and returning a dataset."""

  def __init__(self,
               file_pattern,
               example_secs,
               sample_rate,
               frame_rate,
               data_format_map_fn):
    """RecordProvider constructor."""
    self._file_pattern = file_pattern or self.default_file_pattern
    self._audio_length = example_secs * sample_rate
    self._feature_length = example_secs * frame_rate
    super().__init__(sample_rate, frame_rate)
    self._data_format_map_fn = data_format_map_fn

  @property
  def default_file_pattern(self):
    """Used if file_pattern is not provided to constructor."""
    raise NotImplementedError(
        'You must pass a "file_pattern" argument to the constructor or '
        'choose a FileDataProvider with a default_file_pattern.')

  def get_dataset(self, shuffle=True):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the files.

    Returns:
      dataset: A tf.dataset that reads from the TFRecord.
    """
    def parse_tfexample(record):
      return tf.io.parse_single_example(record, self.features_dict)

    filenames = tf.data.Dataset.list_files(self._file_pattern, shuffle=shuffle)
    dataset = filenames.interleave(
        map_func=self._data_format_map_fn,
        cycle_length=40,
        num_parallel_calls=_AUTOTUNE)
    dataset = dataset.map(parse_tfexample, num_parallel_calls=_AUTOTUNE)
    return dataset

  @property
  def features_dict(self):
    """Dictionary of features to read from dataset."""
    return {
        'env_id':
            tf.io.FixedLenFeature([1], dtype=tf.float32),
        'source_id':
            tf.io.FixedLenFeature([1], dtype=tf.float32),
        'audio':
            tf.io.FixedLenFeature([self._audio_length], dtype=tf.float32),
        'f0_hz':
            tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
        'f0_confidence':
            tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
        'loudness_db':
            tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
    }


@gin.register
class TFRecordProvider(RecordProvider):
  """Class for reading TFRecords and returning a dataset."""

  def __init__(self,
               file_pattern=None,
               example_secs=4,
               sample_rate=16000,
               frame_rate=250):
    """TFRecordProvider constructor."""
    super().__init__(file_pattern, example_secs, sample_rate,
                     frame_rate, tf.data.TFRecordDataset)


# ------------------------------------------------------------------------------
# Zipped DataProvider
# ------------------------------------------------------------------------------
@gin.register
class ZippedProvider(DataProvider):
  """Combines datasets from two providers with zip."""

  def __init__(self, data_providers, batch_size_ratios=()):
    """Constructor.

    Args:
      data_providers: A list of data_providers.
      batch_size_ratios: A list of ratios of batch sizes for each provider.
        These do not need to sum to 1. For example, [2, 1] will produce batches
        with a size ratio of 2 to 1.
    """
    # Normalize the ratios.
    if batch_size_ratios:
      # Check lengths match.
      if len(batch_size_ratios) != len(data_providers):
        raise ValueError('List of batch size ratios ({}) must be of the same '
                         'length as the list of data providers ({}) for varying'
                         'batch sizes.'.format(
                             len(batch_size_ratios), len(data_providers)))
      total = sum(batch_size_ratios)
      batch_size_ratios = [float(bsr) / total for bsr in batch_size_ratios]

    # Make sure all sample rates are the same.
    sample_rates = [dp.sample_rate for dp in data_providers]
    assert len(set(sample_rates)) <= 1
    sample_rate = sample_rates[0]

    # Make sure all frame rates are the same.
    frame_rates = [dp.frame_rate for dp in data_providers]
    assert len(set(frame_rates)) <= 1
    frame_rate = frame_rates[0]

    super().__init__(sample_rate, frame_rate)
    self._data_providers = data_providers
    self._batch_size_ratios = batch_size_ratios

  def get_dataset(self, shuffle=True):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the input files.

    Returns:
      dataset: A zipped tf.data.Dataset from multiple providers.
    """
    datasets = tuple(dp.get_dataset(shuffle) for dp in self._data_providers)
    return tf.data.Dataset.zip(datasets)

  def get_batch(self, batch_size, shuffle=True, repeats=-1):
    """Read dataset.

    Args:
      batch_size: Size of batches, can be a list to have varying batch_sizes.
      shuffle: Whether to shuffle the examples.
      repeats: Number of times to repeat dataset. -1 for endless repeats.

    Returns:
      A batched tf.data.Dataset.
    """
    if not self._batch_size_ratios:
      # One batch size for all datasets ('None' is batch shape).
      return super().get_batch(batch_size)

    else:
      # Varying batch sizes (Integer batch shape for each).
      batch_sizes = [int(batch_size * bsr) for bsr in self._batch_size_ratios]
      datasets = tuple(
          dp.get_dataset(shuffle).batch(bs, drop_remainder=True)
          for bs, dp in zip(batch_sizes, self._data_providers))
      dataset = tf.data.Dataset.zip(datasets)
      dataset = dataset.repeat(repeats)
      dataset = dataset.prefetch(buffer_size=_AUTOTUNE)
      return dataset


# ------------------------------------------------------------------------------
# Synthetic Data for InverseSynthesis
# ------------------------------------------------------------------------------
@gin.register
class SyntheticNotes(TFRecordProvider):
  """Create self-supervised control signal.

  EXPERIMENTAL

  Pass file_pattern to tfrecords created by `ddsp_generate_synthetic_data.py`.
  """

  def __init__(self,
               n_timesteps,
               n_harmonics,
               n_mags,
               file_pattern=None,
               sample_rate=16000):
    self.n_timesteps = n_timesteps
    self.n_harmonics = n_harmonics
    self.n_mags = n_mags
    super().__init__(file_pattern=file_pattern, sample_rate=sample_rate)

  @property
  def features_dict(self):
    """Dictionary of features to read from dataset."""
    return {
        'f0_hz':
            tf.io.FixedLenFeature([self.n_timesteps, 1], dtype=tf.float32),
        'harm_amp':
            tf.io.FixedLenFeature([self.n_timesteps, 1], dtype=tf.float32),
        'harm_dist':
            tf.io.FixedLenFeature(
                [self.n_timesteps, self.n_harmonics], dtype=tf.float32),
        'sin_amps':
            tf.io.FixedLenFeature(
                [self.n_timesteps, self.n_harmonics], dtype=tf.float32),
        'sin_freqs':
            tf.io.FixedLenFeature(
                [self.n_timesteps, self.n_harmonics], dtype=tf.float32),
        'noise_magnitudes':
            tf.io.FixedLenFeature(
                [self.n_timesteps, self.n_mags], dtype=tf.float32),
    }

@gin.register
class AudioProvider(DataProvider):
  """Class for reading wav files and returning a dataset."""

  def __init__(self,
               file_pattern=None,
               n_samples=64000,
               audio_sample_rate=16000,
               frame_rate=250,
               append_material_id=False):
    self.random_generator = tf.random.Generator.from_seed(1)

    self._file_pattern = file_pattern or self.default_file_pattern
    self._audio_length = n_samples
    self._feature_length = int(n_samples / audio_sample_rate * frame_rate)
    super().__init__(audio_sample_rate, frame_rate)
    self.append_material_id = append_material_id
    if self.append_material_id:
      self.video_table, self.material_table = self.get_tables(self._file_pattern)
      print(f"Size of video table: {self.video_table.shape}, size of material table: {self.material_table.shape}")
      print(f"Material ID table: {self.material_table}")
    
    def map_fn(filename):
      audio = tf.io.read_file(filename)
      decoded_audio, audio_sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
      audio_frame_count = tf.shape(decoded_audio)[0]
      decoded_audio = tf.expand_dims(tf.squeeze(decoded_audio), axis=0)
      start_index = self.random_generator.uniform([], minval=0, maxval=(audio_frame_count - self._audio_length) + 1, dtype=tf.dtypes.int32)
      clipped_audio = decoded_audio[:, start_index:(start_index + self._audio_length)]
      if self.append_material_id:
        f = tf.strings.split(filename, '/')[-1]
        video_id = tf.strings.split(f, '-')[0]
        material_id = tf.strings.split(f, '-')[1]
        material_id = self.material_table.lookup(tf.strings.join((video_id, material_id), '-'))
        material_id = tf.expand_dims(material_id, axis=0)
        print(f"material_id: {material_id} and video_id: {video_id}")
        video_id = self.video_table.lookup(video_id)
        video_id = tf.expand_dims(video_id, axis=0)
        return tf.data.Dataset.from_tensor_slices({'audio':clipped_audio, 'video_id':[video_id], 'material_id':[material_id], 'filename': [filename]})
      return tf.data.Dataset.from_tensor_slices({'audio':clipped_audio})

    self._data_format_map_fn = map_fn

  @staticmethod
  def get_tables(file_pattern):
    file_paths = glob.glob(file_pattern)
    filenames = [os.path.split(f)[1] for f in file_paths]
    video_ids = set([f.split('-')[0] for f in filenames])
    material_ids = set(['-'.join(f.split('-')[:2]) for f in filenames])
    video_keys_tensor = tf.constant(sorted(list(video_ids)), dtype=tf.string)
    video_values_tensor = tf.range(len(video_ids), dtype=tf.int32)
    video_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(video_keys_tensor, video_values_tensor), -1)
    material_keys_tensor = tf.constant(sorted(list(material_ids)), dtype=tf.string)
    material_values_tensor = tf.range(len(material_ids), dtype=tf.int32)
    material_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(material_keys_tensor, material_values_tensor), -1)
    return video_table, material_table


  @property
  def default_file_pattern(self):
    """Used if file_pattern is not provided to constructor."""
    raise NotImplementedError(
        'You must pass a "file_pattern" argument to the constructor or '
        'choose a FileDataProvider with a default_file_pattern.')

  def get_dataset(self, shuffle=True):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the files.

    Returns:
      dataset: A tf.dataset that reads from the TFRecord.
    """
    filenames = tf.data.Dataset.list_files(self._file_pattern, shuffle=shuffle)
    dataset = filenames.interleave(
        map_func=self._data_format_map_fn,
        cycle_length=40,
        num_parallel_calls=_AUTOTUNE)
    return dataset

@gin.register
class VideoProvider(AudioProvider):
  """Class for reading video records and returning a dataset."""

  def __init__(self,
               file_pattern=None,
               example_secs=4,
               audio_sample_rate=16000,
               append_material_id=False,
               video_frame_rate=30,
               frame_rate=250):
    super().__init__( file_pattern=file_pattern,
                  n_samples=example_secs * audio_sample_rate,
                  audio_sample_rate=audio_sample_rate,
                  frame_rate=frame_rate,
                  append_material_id=append_material_id)
    self._video_length = int(example_secs * video_frame_rate)
    self._video_padding_frames = 1

    def map_fn(filename):
      audio_filename = tf.strings.join([tf.strings.split(filename, '.')[0], 'wav'], '.')
      audio = tf.io.read_file(audio_filename)
      decoded_audio, audio_sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
      decoded_audio = tf.expand_dims(tf.squeeze(decoded_audio), axis=0)
      video = tf.io.read_file(filename)
      decoded_video = tfio.experimental.ffmpeg.decode_video(video)
      video_frame_count = tf.shape(decoded_video)[0]
      start_index = self.random_generator.uniform([], minval=0, maxval=(video_frame_count - self._video_length - self._video_padding_frames), dtype=tf.dtypes.int32)
      clipped_video = decoded_video[start_index:(start_index + self._video_length), ...]
      clipped_video = tf.expand_dims(clipped_video, axis=0)
      audio_start = tf.cast(start_index * self._audio_length / self._video_length, dtype=tf.int32)
      clipped_audio = decoded_audio[:, audio_start:(audio_start + self._audio_length)]
      return tf.data.Dataset.from_tensor_slices({'frames':clipped_video, 'audio':clipped_audio})

    self._data_format_map_fn = map_fn

@gin.register
class VISProvider(DataProvider):
  """Class for reading video records and returning a dataset."""

  def __init__(self,
               data_dir=None,
               test_mode=False,
               event_filter='hit',
               examples_per_clip=1,
               example_secs=0.5,
               audio_sample_rate=96000,
               video_frame_rate=30,
               frame_rate=250):
    super().__init__(audio_sample_rate, frame_rate)
    video_length = int(example_secs * video_frame_rate)
    audio_length = int(1.0 * video_length / video_frame_rate * audio_sample_rate)
    self.test_mode = test_mode
    self.data_dir = data_dir

    def map_fn(filename):
      metadata = tf.io.read_file(filename + '_times.txt')
      metadata = tf.strings.split(tf.strings.split(metadata, sep='\n'), sep=' ')
      event_times = tf.strings.to_number(metadata[:-1, :1], out_type=tf.float32)
      event_times = event_times.to_tensor()
      event_times = tf.squeeze(event_times)
      if event_filter:
        mask = tf.equal(metadata[:, 2:3], event_filter).to_tensor()
        mask = tf.reshape(mask, [-1])
        event_times = tf.boolean_mask(event_times, mask)
      event_indices = tf.random.uniform([], maxval=tf.shape(event_times)[0], dtype=tf.int32)
      event_times = tf.gather(event_times, event_indices)

      video = tf.io.read_file(filename + '_denoised_thumb.mp4')
      decoded_video = tfio.experimental.ffmpeg.decode_video(video)
      video_start_index = tf.cast((event_times - (example_secs / 2.0)) * video_frame_rate, tf.int32)
      clipped_video = decoded_video[video_start_index:(video_start_index + video_length), ...]
      clipped_video = tf.expand_dims(clipped_video, axis=0)

      audio = tf.io.read_file(filename + '_denoised.wav')
      decoded_audio, metadata_audio_sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
      decoded_audio = tf.expand_dims(tf.squeeze(decoded_audio), axis=0)
      audio_start_index = tf.cast(video_start_index / video_frame_rate * audio_sample_rate, tf.int32)
      clipped_audio = decoded_audio[:, audio_start_index:(audio_start_index+audio_length)]
      return tf.data.Dataset.from_tensor_slices({'frames':clipped_video, 'audio':clipped_audio})

    self._data_format_map_fn = map_fn

  def get_dataset(self, shuffle=True):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the files.

    Returns:
      dataset: A tf.dataset that reads from the TFRecord.
    """
    if self.test_mode:
      file_list = tf.io.read_file(os.path.join(self.data_dir, 'test.txt'))
    else:
      file_list = tf.io.read_file(os.path.join(self.data_dir, 'train.txt'))
    file_list = tf.strings.split(file_list, sep='\n')[:-1]
    filenames = os.path.join(self.data_dir, '') + file_list
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
      filenames = filenames.shuffle(10000)
    dataset = filenames.interleave(
        map_func=self._data_format_map_fn,
        cycle_length=10,
        num_parallel_calls=_AUTOTUNE)
    return dataset
