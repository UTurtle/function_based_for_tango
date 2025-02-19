# noise_pipeline/spectrogram_modifier.py

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class SpectrogramModifier:
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        window='hann',
        noise_strength=0.1,
        noise_type='normal',
        noise_params=None
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window= window
        self.noise_strength = noise_strength
        self.noise_type = noise_type
        self.noise_params = noise_params if noise_params else {}
        self.signal = None
        self.signal_with_noise = None
        self.S_db = None

    def _generate_normal_noise(self, length, params):
        mean = params.get('mean', 0.0)
        std = params.get('std', 1.0)
        return np.random.normal(mean, std, length)

    def _generate_uniform_noise(self, length, params):
        low = params.get('low', -1.0)
        high = params.get('high', 1.0)
        return np.random.uniform(low, high, length)

    def _generate_perlin_noise(self, length, params):
        def fade(t):
            return 6 * t**5 - 15 * t**4 + 10 * t**3

        seed = params.get('seed', 42)
        np.random.seed(seed)
        perm = np.arange(256)
        np.random.shuffle(perm)
        perm = np.stack([perm, perm]).flatten()
        scale = params.get('scale', 50.0)
        xs = np.linspace(0, length / scale, length)
        xi = np.floor(xs).astype(int)
        xf = xs - xi
        xi = xi % 256
        left_hash = perm[xi]
        right_hash = perm[xi + 1]
        u = fade(xf)
        left_grad = ((left_hash & 1) * 2 - 1) * xf
        right_grad = ((right_hash & 1) * 2 - 1) * (xf - 1)
        noise = (1 - u) * left_grad + u * right_grad
        noise = noise / np.max(np.abs(noise))
        return noise

    def generate_noise(self, signal):
        length = len(signal)
        noise_type = self.noise_type
        params = self.noise_params
        seed = params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        if noise_type == 'normal':
            noise = self._generate_normal_noise(length, params)
        elif noise_type == 'uniform':
            noise = self._generate_uniform_noise(length, params)
        elif noise_type == 'perlin':
            noise = self._generate_perlin_noise(length, params)
        else:
            noise = np.zeros_like(signal)
        return signal + noise * self.noise_strength

    def compute_spectrogram(self, signal):
        self.signal = signal
        self.signal_with_noise = self.generate_noise(signal)
        S = np.abs(
            librosa.stft(
                self.signal_with_noise,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window
            )
        )
        self.S_db = librosa.amplitude_to_db(S, ref=np.max)
        return self.S_db

    def _get_freqs(self):
        return np.linspace(0, self.sample_rate / 2, self.S_db.shape[0])

    def _get_times(self):
        return librosa.frames_to_time(
            np.arange(self.S_db.shape[1]),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

    def apply_dB_mask(self, dB_mask):
        self.S_db += dB_mask

    def plot_spectrogram(self, show_labels=True, colormap='magma', title='Spectrogram'):
        if self.S_db is None:
            raise ValueError("compute_spectrogram() must be called first.")
        fig, ax = plt.subplots(dpi=100)
        img = librosa.display.specshow(
            self.S_db,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='linear',
            ax=ax,
            cmap=colormap
        )
        if show_labels:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(title)
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
        else:
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.tight_layout(pad=0.5)
        return fig, ax
