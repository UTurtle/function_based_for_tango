# noise_pipeline/shapes.py

import numpy as np
from abc import ABC, abstractmethod
from matplotlib.path import Path


class DBShape(ABC):
    @abstractmethod
    def create_mask(self, spectro_shape, spectro_mod):
        pass


class BaseShape(DBShape):
    def __init__(self):
        pass

    @abstractmethod
    def generate_shape_mask(self, spectro_shape, spectro_mod):
        pass

    def create_mask(self, spectro_shape, spectro_mod):
        return self.generate_shape_mask(spectro_shape, spectro_mod)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


class CircleDBShape(BaseShape):
    def __init__(self, center_freq, center_time, radius_freq, radius_time, strength_dB):
        super().__init__()
        self.center_freq = center_freq
        self.center_time = center_time
        self.radius_freq = radius_freq
        self.radius_time = radius_time
        self.strength_dB = strength_dB

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        freqs = spectro_mod._get_freqs()
        times = spectro_mod._get_times()
        ff, tt = np.meshgrid(freqs, times, indexing='ij')
        dist = (
            (ff - self.center_freq) ** 2 / (self.radius_freq ** 2)
            + (tt - self.center_time) ** 2 / (self.radius_time ** 2)
        )
        circle = (dist <= 1).astype(float)
        return circle * self.strength_dB


class TrapezoidDBShape(BaseShape):
    def __init__(
        self,
        freq_min,
        freq_max,
        time_min,
        time_max,
        slope_freq,
        slope_time,
        strength_dB
    ):
        super().__init__()
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.time_min = time_min
        self.time_max = time_max
        self.slope_freq = slope_freq
        self.slope_time = slope_time
        self.strength_dB = strength_dB

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        mask = np.zeros(spectro_shape)
        freqs = spectro_mod._get_freqs()
        times = spectro_mod._get_times()

        freq_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        time_mask = (times >= self.time_min) & (times <= self.time_max)
        f_inds = np.where(freq_mask)[0]
        t_inds = np.where(time_mask)[0]

        if len(f_inds) == 0 or len(t_inds) == 0:
            return mask

        f_dist = (freqs[f_inds] - self.freq_min) / (self.freq_max - self.freq_min)
        t_dist = (times[t_inds] - self.time_min) / (self.time_max - self.time_min)

        for i, fi in enumerate(f_inds):
            for j, ti in enumerate(t_inds):
                val = self.strength_dB
                val *= (1 - abs(f_dist[i] - 0.5) * 2 * self.slope_freq)
                val *= (1 - abs(t_dist[j] - 0.5) * 2 * self.slope_time)
                mask[fi, ti] += val
        return mask


class SpikeDBShape(BaseShape):
    def __init__(
        self,
        center_freq,
        center_time,
        radius_freq,
        radius_time,
        strength_dB,
        rotate=0.0
    ):
        super().__init__()
        self.center_freq = center_freq
        self.center_time = center_time
        self.radius_freq = radius_freq
        self.radius_time = radius_time
        self.strength_dB = strength_dB
        self.rotate_deg = rotate

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        freqs = spectro_mod._get_freqs()
        times = spectro_mod._get_times()
        ff, tt = np.meshgrid(freqs, times, indexing='ij')

        f_shift = ff - self.center_freq
        t_shift = tt - self.center_time

        angle_rad = np.deg2rad(self.rotate_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        f_rot = f_shift * cos_a - t_shift * sin_a
        t_rot = f_shift * sin_a + t_shift * cos_a

        dist = np.sqrt(
            (f_rot**2) / (self.radius_freq ** 2) +
            (t_rot**2) / (self.radius_time ** 2)
        )
        spike = np.exp(-dist**2) * self.strength_dB
        return spike


class PillarDBShape(BaseShape):
    def __init__(self, freq_min, freq_max, strength_dB):
        super().__init__()
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.strength_dB = strength_dB

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        mask = np.zeros(spectro_shape)
        freqs = spectro_mod._get_freqs()
        freq_indices = np.where(
            (freqs >= self.freq_min) & (freqs <= self.freq_max)
        )[0]
        if len(freq_indices) == 0:
            return mask
        mask[freq_indices, :] += self.strength_dB
        return mask


class RectangleDBShape(BaseShape):
    def __init__(
        self,
        freq_min,
        freq_max,
        time_min,
        time_max,
        strength_dB
    ):
        super().__init__()
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.time_min = time_min
        self.time_max = time_max
        self.strength_dB = strength_dB

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        mask = np.zeros(spectro_shape)
        freqs = spectro_mod._get_freqs()
        times = spectro_mod._get_times()
        freq_indices = np.where(
            (freqs >= self.freq_min) & (freqs <= self.freq_max)
        )[0]
        time_indices = np.where(
            (times >= self.time_min) & (times <= self.time_max)
        )[0]
        if len(freq_indices) == 0 or len(time_indices) == 0:
            return mask
        mask[np.ix_(freq_indices, time_indices)] += self.strength_dB
        return mask


class EllipseDBShape(BaseShape):
    def __init__(
        self,
        center_freq,
        center_time,
        radius_freq,
        radius_time,
        strength_dB
    ):
        super().__init__()
        self.center_freq = center_freq
        self.center_time = center_time
        self.radius_freq = radius_freq
        self.radius_time = radius_time
        self.strength_dB = strength_dB

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        freqs = spectro_mod._get_freqs()
        times = spectro_mod._get_times()
        ff, tt = np.meshgrid(freqs, times, indexing='ij')
        dist = (
            (ff - self.center_freq) ** 2 / (self.radius_freq ** 2) +
            (tt - self.center_time) ** 2 / (self.radius_time ** 2)
        )
        ellipse = (dist <= 1).astype(float)
        return ellipse * self.strength_dB


class HorizontalSpikeDBShape(BaseShape):
    def __init__(
        self,
        center_freq,
        center_time,
        radius_freq,
        radius_time,
        strength_dB
    ):
        super().__init__()
        self.center_freq = center_freq
        self.center_time = center_time
        self.radius_freq = radius_freq
        self.radius_time = radius_time
        self.strength_dB = strength_dB

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        spike_shape = SpikeDBShape(
            center_freq=self.center_freq,
            center_time=self.center_time,
            radius_freq=self.radius_freq,
            radius_time=self.radius_time,
            strength_dB=self.strength_dB,
            rotate=0.0
        )
        return spike_shape.generate_shape_mask(spectro_shape, spectro_mod)


class VerticalSpikeDBShape(BaseShape):
    def __init__(
        self,
        center_freq,
        center_time,
        radius_freq,
        radius_time,
        strength_dB
    ):
        super().__init__()
        self.center_freq = center_freq
        self.center_time = center_time
        self.radius_freq = radius_freq
        self.radius_time = radius_time
        self.strength_dB = strength_dB

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        spike_shape = SpikeDBShape(
            center_freq=self.center_freq,
            center_time=self.center_time,
            radius_freq=self.radius_freq,
            radius_time=self.radius_time,
            strength_dB=self.strength_dB,
            rotate=0.0
        )
        return spike_shape.generate_shape_mask(spectro_shape, spectro_mod)


class HorizontalRangeDistributionDBShape(BaseShape):
    def __init__(
        self,
        freq_min,
        freq_max,
        strength_dB,
        distribution='gaussian',
        distribution_params=None
    ):
        super().__init__()
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.strength_dB = strength_dB
        self.distribution = distribution
        self.distribution_params = distribution_params if distribution_params else {}

    def _get_distribution(self, values):
        sigma = self.distribution_params.get('sigma', 1000)
        center = (values.min() + values.max()) / 2
        dist = np.exp(-0.5 * ((values - center) / sigma) ** 2)
        if dist.max() != 0:
            dist /= dist.max()
        return dist

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        mask = np.zeros(spectro_shape)
        freqs = spectro_mod._get_freqs()
        freq_indices = np.where(
            (freqs >= self.freq_min) & (freqs <= self.freq_max)
        )[0]

        if len(freq_indices) == 0:
            return mask

        freq_values = freqs[freq_indices]
        dist_values = self._get_distribution(freq_values)

        for i, fi in enumerate(freq_indices):
            mask[fi, :] += self.strength_dB * dist_values[i]

        return mask


class VerticalRangeDistributionDBShape(BaseShape):
    def __init__(
        self,
        time_min,
        time_max,
        strength_dB,
        distribution='gaussian',
        distribution_params=None
    ):
        super().__init__()
        self.time_min = time_min
        self.time_max = time_max
        self.strength_dB = strength_dB
        self.distribution = distribution
        self.distribution_params = distribution_params if distribution_params else {}

    def _get_distribution(self, values):
        sigma = self.distribution_params.get('sigma', 0.5)
        center = (values.min() + values.max()) / 2
        dist = np.exp(-0.5 * ((values - center) / sigma) ** 2)
        dist /= dist.max()
        return dist

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        mask = np.zeros(spectro_shape)
        times = spectro_mod._get_times()
        time_indices = np.where(
            (times >= self.time_min) & (times <= self.time_max)
        )[0]
        if len(time_indices) == 0:
            return mask

        time_vals = times[time_indices]
        dist_values = self._get_distribution(time_vals)

        for i, ti in enumerate(time_indices):
            mask[:, ti] += self.strength_dB * dist_values[i]

        return mask


class HillDBShape(BaseShape):
    def __init__(
        self,
        freq_center,
        time_center,
        freq_width,
        time_width,
        strength_dB
    ):
        super().__init__()
        self.freq_center = freq_center
        self.time_center = time_center
        self.freq_width = freq_width
        self.time_width = time_width
        self.strength_dB = strength_dB

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        freqs = spectro_mod._get_freqs()
        times = spectro_mod._get_times()
        ff, tt = np.meshgrid(freqs, times, indexing='ij')
        dist = np.sqrt(
            ((ff - self.freq_center) ** 2) / (self.freq_width ** 2) +
            ((tt - self.time_center) ** 2) / (self.time_width ** 2)
        )
        hill = (1 - dist)
        hill[hill < 0] = 0
        return hill * self.strength_dB


class FogDBShape(BaseShape):
    def __init__(self, strength_dB, coverage=1.0):
        super().__init__()
        self.strength_dB = strength_dB
        self.coverage = coverage

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        random_map = np.random.uniform(0, 1, spectro_shape)
        fog = (random_map < self.coverage).astype(float)
        fog *= self.strength_dB * (np.random.randn(*spectro_shape) * 0.1)
        return fog


class PolygonDBShape(BaseShape):
    def __init__(self, vertices, strength_dB):
        super().__init__()
        self.vertices = vertices
        self.strength_dB = strength_dB

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        freqs = spectro_mod._get_freqs()
        times = spectro_mod._get_times()
        ff, tt = np.meshgrid(freqs, times, indexing='ij')

        path = Path(self.vertices)
        points = np.vstack((ff.ravel(), tt.ravel())).T
        inside = path.contains_points(points).reshape(ff.shape)
        mask = np.zeros(spectro_shape)
        mask[inside] = self.strength_dB
        return mask


class WavePatternDBShape(BaseShape):
    def __init__(self, axis='time', frequency=1.0, strength_dB=5.0):
        super().__init__()
        self.axis = axis
        self.frequency = frequency
        self.strength_dB = strength_dB

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        freqs = spectro_mod._get_freqs()
        times = spectro_mod._get_times()
        mask = np.zeros(spectro_shape)
        if self.axis == 'time':
            wave = np.sin(2 * np.pi * self.frequency * times)
            mask += wave[np.newaxis, :] * self.strength_dB
        else:
            wave = np.sin(2 * np.pi * self.frequency * freqs)
            mask += wave[:, np.newaxis] * self.strength_dB
        return mask


class RealWorldNoiseDBShape(BaseShape):
    def __init__(
        self,
        audio_path,
        freq_min,
        freq_max,
        time_min,
        time_max,
        strength_dB=0
    ):
        super().__init__()
        self.audio_path = audio_path
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.time_min = time_min
        self.time_max = time_max
        self.strength_dB = strength_dB

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        y, sr = librosa.load(self.audio_path, sr=spectro_mod.sample_rate)
        S = np.abs(
            librosa.stft(y, n_fft=spectro_mod.n_fft, hop_length=spectro_mod.hop_length)
        )
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        freqs = spectro_mod._get_freqs()
        times = spectro_mod._get_times()
        freq_indices = np.where(
            (freqs >= self.freq_min) & (freqs <= self.freq_max)
        )[0]
        time_indices = np.where(
            (times >= self.time_min) & (times <= self.time_max)
        )[0]

        if len(freq_indices) == 0 or len(time_indices) == 0:
            return np.zeros(spectro_shape)

        f_cut = min(len(freq_indices), S_db.shape[0])
        t_cut = min(len(time_indices), S_db.shape[1])

        mask = np.zeros(spectro_shape)
        mask[freq_indices[:f_cut], time_indices[:t_cut]] += (
            S_db[:f_cut, :t_cut] + self.strength_dB
        )
        return mask


class HorizontalLineDBShape(BaseShape):
    def __init__(self, center_freq, strength_dB, thickness=2):
        super().__init__()
        self.center_freq = center_freq
        self.strength_dB = strength_dB
        self.thickness = thickness

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        mask = np.zeros(spectro_shape)
        freqs = spectro_mod._get_freqs()
        freq_idx = np.argmin(np.abs(freqs - self.center_freq))
        start_idx = max(freq_idx - self.thickness // 2, 0)
        end_idx = min(freq_idx + self.thickness // 2 + 1, spectro_shape[0])
        mask[start_idx:end_idx, :] += self.strength_dB
        return mask


class VerticalLineDBShape(BaseShape):
    def __init__(self, center_time, strength_dB, thickness=2):
        super().__init__()
        self.center_time = center_time
        self.strength_dB = strength_dB
        self.thickness = thickness

    def generate_shape_mask(self, spectro_shape, spectro_mod):
        mask = np.zeros(spectro_shape)
        times = spectro_mod._get_times()
        time_idx = np.argmin(np.abs(times - self.center_time))
        start_idx = max(time_idx - self.thickness // 2, 0)
        end_idx = min(time_idx + self.thickness // 2 + 1, spectro_shape[1])
        mask[:, start_idx:end_idx] += self.strength_dB
        return mask
