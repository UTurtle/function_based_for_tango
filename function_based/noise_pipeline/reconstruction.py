# noise_pipeline/reconstruction.py

import os
import sys
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.ndimage import gaussian_filter

from .spectrogram_modifier import SpectrogramModifier
from .noise_pipeline import NoisePipeline
from .factories import ShapeFactory, PatternFactory

# Existing functions
def reconstruct_with_griffinlim(
        spectrogram_db, 
        hop_length,
        n_iter=32, 
        window='hann'
    ):
    """
    Reconstruct audio from a spectrogram using the Griffin-Lim algorithm.
    
    Parameters:
    - spectrogram_db (np.ndarray): Spectrogram in decibels.
    - hop_length (int): Number of samples between successive frames.
    - n_iter (int): Number of iterations for Griffin-Lim.
    - window (str): Type of window function.
    
    Returns:
    - new_audio (np.ndarray): Reconstructed audio signal.
    """
    amplitude = librosa.db_to_amplitude(spectrogram_db, ref=1.0)
    new_audio = librosa.griffinlim(
        amplitude,
        n_iter=n_iter,
        hop_length=hop_length,
        window=window
    )
    return new_audio


def reconstruct_audio_from_final_spectrogram(spectro_mod):
    """
    Reconstruct audio from the final spectrogram using the phase of the original signal.
    
    Parameters:
    - spectro_mod (SpectrogramModifier): Instance of SpectrogramModifier with computed spectrogram.
    
    Returns:
    - new_audio (np.ndarray): Reconstructed audio signal.
    """
    stft_orig = librosa.stft(
        spectro_mod.signal_with_noise,
        n_fft=spectro_mod.n_fft,
        hop_length=spectro_mod.hop_length,
        window=spectro_mod.window
    )
    phase = np.angle(stft_orig)
    magnitude = librosa.db_to_amplitude(spectro_mod.S_db, ref=1.0)
    new_stft = magnitude * np.exp(1j * phase)

    new_audio = librosa.istft(
        new_stft,
        hop_length=spectro_mod.hop_length,
        window=spectro_mod.window
    )
    return new_audio

# New integrated functionality

def reconstruct_pipeline_spectrogram_and_audio(parsed_explanation):
    """
    Reconstruct the spectrogram and audio based on parsed shapes and patterns.

    Parameters:
    - parsed_explanation (dict): Dictionary containing spectrogram base settings,
                                 shapes, and patterns.

    Returns:
    - S_db (np.ndarray): The final spectrogram in dB.
    - reconstructed_audio (np.ndarray): The reconstructed audio signal.
    - sample_rate (int): The sample rate of the audio.
    """
    spectrogram_base = parsed_explanation.get('spectrogram_base', {})
    shapes = parsed_explanation.get('shapes', [])
    patterns = parsed_explanation.get('patterns', [])

    sample_rate = spectrogram_base.get('sample_rate', 16000)
    n_fft = spectrogram_base.get('n_fft', 1024)
    hop_length = spectrogram_base.get('hop_length', 512)
    noise_strength = spectrogram_base.get('noise_strength', 0.1)
    noise_type = spectrogram_base.get('noise_type', 'normal')
    noise_params = spectrogram_base.get('noise_params', {})

    # Initialize SpectrogramModifier
    spectro_mod = SpectrogramModifier(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        noise_strength=noise_strength,
        noise_type=noise_type,
        noise_params=noise_params
    )

    # Initialize NoisePipeline
    pipeline = NoisePipeline(
        spectro_mod=spectro_mod,
        apply_blur=False,
        blur_sigma=1.0
    )

    # Add shapes and patterns
    shape_factory = ShapeFactory()
    pattern_factory = PatternFactory()

    for shape_info in shapes:
        shape_name = shape_info.get('type', '').lower()
        shape_params = shape_info.get('parameters', {})
        try:
            shape_obj = shape_factory.create(shape_name, **shape_params)
            pipeline.add_shape(shape_obj)
        except Exception as e:
            print(f"Error creating shape '{shape_name}': {e}")

    for pattern_info in patterns:
        pattern_name = pattern_info.get('type', '').lower()
        pattern_params = pattern_info.get('parameters', {})
        try:
            pattern_obj = pattern_factory.create(pattern_name, pattern_params)
            pipeline.add_pattern(pattern_obj)
        except Exception as e:
            print(f"Error creating pattern '{pattern_name}': {e}")

    # Generate spectrogram and reconstruct audio
    duration = parsed_explanation.get('duration', 12.0)  # Default to 12.0 seconds
    signal_length = int(sample_rate * duration)
    silence_signal = np.zeros(signal_length)

    pipeline.generate(silence_signal)
    S_db = spectro_mod.S_db.copy()
    reconstructed_audio = reconstruct_audio_from_final_spectrogram(spectro_mod)

    return S_db, reconstructed_audio, sample_rate


def save_and_visualize_results(S_db, audio, sr, output_dir, file_id):
    """
    Save and visualize spectrogram and audio.

    Parameters:
    - S_db (np.ndarray): Spectrogram in dB.
    - audio (np.ndarray): Audio signal.
    - sr (int): Sample rate.
    - output_dir (str): Directory to save outputs.
    - file_id (str): Identifier for the output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='linear', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram: {file_id}")
    spectrogram_path = os.path.join(output_dir, f"{file_id}_spectrogram.png")
    plt.savefig(spectrogram_path, dpi=100)
    plt.close()

    # Save audio
    audio_path = os.path.join(output_dir, f"{file_id}.wav")
    sf.write(audio_path, audio, sr)

    print(f"Spectrogram saved at: {spectrogram_path}")
    print(f"Audio saved at: {audio_path}")


def load_parsed_explanation(json_path):
    """
    Load parsed explanation from a JSON file.

    Parameters:
    - json_path (str): Path to the JSON file.

    Returns:
    - parsed_explanation (dict): Parsed JSON data.
    """
    with open(json_path, 'r') as f:
        parsed_explanation = json.load(f)
    return parsed_explanation


def reconstruct_from_json(json_input_path, output_dir, file_id):
    """
    Reconstruct spectrogram and audio from a JSON description.

    Parameters:
    - json_input_path (str): Path to the JSON input file.
    - output_dir (str): Directory to save the output spectrogram and audio.
    - file_id (str): Identifier for the output files.
    """
    if not os.path.exists(json_input_path):
        print(f"JSON input file does not exist: {json_input_path}")
        return

    parsed_explanation = load_parsed_explanation(json_input_path)
    S_db, reconstructed_audio, sample_rate = reconstruct_pipeline_spectrogram_and_audio(parsed_explanation)
    save_and_visualize_results(S_db, reconstructed_audio, sample_rate, output_dir, file_id)


