import os
import glob
import librosa
import numpy as np
import tensorflow as tf


def get_wav_files(path):
    """
    Get a list of WAV file names from the specified path.

    Parameters:
    - path (str): Path to the directory containing WAV files.

    Returns:
    - List[str]: List of WAV file names.
    """
    return [os.path.basename(x) for x in glob.glob(path + '/*.wav')]


def get_graph_size():
    """
    Calculate and return the size of the TensorFlow graph in megabytes.

    Returns:
    - str: Size of the TensorFlow graph in megabytes.
    """
    vars = 0
    for v in tf.all_variables():
        vars += np.prod(v.get_shape().as_list())
    return "{} mega".format((4 * vars) / (1e6))
    

def config_dataset(dataset, flags):
    """
    Configure the input dataset for training.

    Parameters:
    - dataset: TensorFlow dataset.
    - flags: Flags object containing configuration parameters.

    Returns:
    - dataset: Configured TensorFlow dataset.
    """
    dataset = dataset.repeat(flags.epochs)
    dataset = dataset.batch(flags.batch_size)
    dataset = dataset.prefetch(flags.batch_size)
    return dataset


def audio_generator_complex(files, flags):
    """
    Generate audio samples for training or testing.

    Parameters:
    - files (list): List of WAV file names.
    - flags: Flags object containing configuration parameters.

    Yields:
    - tuple: Tuple containing noise and clean audio samples.
    """

    for file in files:
        noise, fs = librosa.load(os.path.join(flags.train_noise_dir, file), sr=flags.fs)
        clean, fs = librosa.load(os.path.join(flags.train_clean_dir, file), sr=flags.fs)

        channels = flags.channels
        window = int(channels)
        size_samples = len(noise)
        assert len(noise) == len(clean), "clean and noise lengths must match"

        samples_per_file = int((size_samples // window))

        for i in range(size_samples):
            if i * window + channels >= size_samples:
                break
            noise_channels_raw = noise[i * window:i * window + channels]
            clean_channels_raw = clean[i * window:i * window + channels]

            yield noise_channels_raw, clean_channels_raw, samples_per_file
