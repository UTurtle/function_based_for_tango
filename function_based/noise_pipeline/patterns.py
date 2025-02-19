# noise_pipeline/patterns.py

from abc import ABC, abstractmethod
import numpy as np

class Pattern(ABC):
    @abstractmethod
    def create_mask(self, spectro_shape, spectro_mod, shape_factory):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


class LinearPattern(Pattern):
    def __init__(
        self,
        shape_name,
        shape_params,
        direction='time',
        repeat=5,
        spacing=1.0
    ):
        self.shape_name = shape_name
        self.shape_params = shape_params
        self.direction = direction
        self.repeat = repeat
        self.spacing = spacing

    def create_mask(self, spectro_shape, spectro_mod, shape_factory):
        mask = np.zeros(spectro_shape)
        for i in range(self.repeat):
            offset = i * self.spacing
            params = self.shape_params.copy()
            if self.direction == 'time' and 'center_time' in params:
                params['center_time'] += offset
            elif self.direction == 'freq' and 'center_freq' in params:
                params['center_freq'] += offset

            shape_obj = shape_factory.create(self.shape_name, **params)
            if shape_obj is not None:
                mask += shape_obj.create_mask(spectro_shape, spectro_mod)
        return mask


class RandomPattern(Pattern):
    def __init__(
        self,
        shape_name,
        shape_params,
        n=10,
        freq_range=None,
        time_range=None
    ):
        self.shape_name = shape_name
        self.shape_params = shape_params
        self.n = n
        self.freq_range = freq_range if freq_range else (0, 8000)
        self.time_range = time_range if time_range else (0, 10)

    def create_mask(self, spectro_shape, spectro_mod, shape_factory):
        mask = np.zeros(spectro_shape)
        for _ in range(self.n):
            params = self.shape_params.copy()
            if 'center_freq' in params:
                params['center_freq'] = np.random.uniform(*self.freq_range)
            if 'center_time' in params:
                params['center_time'] = np.random.uniform(*self.time_range)

            shape_obj = shape_factory.create(self.shape_name, **params)
            if shape_obj:
                mask += shape_obj.create_mask(spectro_shape, spectro_mod)
        return mask


class NLinearRepeatTSleepPattern(Pattern):
    def __init__(
        self,
        shape_name,
        shape_params,
        repeat=3,
        repeat_time=0.5,
        repeat_hz=None,
        sleep_time=5.0,
        start_time=0.0,
        direction='time'
    ):
        self.shape_name = shape_name
        self.shape_params = shape_params
        self.repeat = repeat
        self.repeat_time = repeat_time
        self.repeat_hz = repeat_hz if repeat_hz is not None else repeat_time * 1000
        self.sleep_time = sleep_time
        self.start_time = start_time
        self.direction = direction

    def create_mask(self, spectro_shape, spectro_mod, shape_factory):
        mask = np.zeros(spectro_shape)
        if self.direction == 'time':
            current_time = self.start_time
        elif self.direction == 'freq':
            current_freq = self.start_time
        else:
            raise ValueError("Invalid direction. Use 'time' or 'freq'.")

        for _ in range(self.repeat):
            params = self.shape_params.copy()
            if self.direction == 'time' and 'center_time' in params:
                params['center_time'] = current_time
            elif self.direction == 'freq' and 'center_freq' in params:
                params['center_freq'] = current_freq

            shape_obj = shape_factory.create(self.shape_name, **params)
            if shape_obj:
                mask += shape_obj.create_mask(spectro_shape, spectro_mod)

            if self.direction == 'time':
                current_time += self.repeat_time
            elif self.direction == 'freq':
                current_freq += self.repeat_hz

        return mask


class ConvexPattern(Pattern):
    def __init__(
        self,
        shape_name,
        shape_params,
        freq_min,
        freq_max,
        time_min,
        time_max,
        n=10
    ):
        self.shape_name = shape_name
        self.shape_params = shape_params
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.time_min = time_min
        self.time_max = time_max
        self.n = n

    def create_mask(self, spectro_shape, spectro_mod, shape_factory):
        mask = np.zeros(spectro_shape)
        freqs = np.linspace(self.freq_min, self.freq_max, self.n)
        times = np.linspace(self.time_min, self.time_max, self.n)
        for i in range(self.n):
            params = self.shape_params.copy()
            if 'center_freq' in params:
                params['center_freq'] = freqs[i]
            if 'center_time' in params:
                params['center_time'] = times[i]

            shape_obj = shape_factory.create(self.shape_name, **params)
            if shape_obj:
                mask += shape_obj.create_mask(spectro_shape, spectro_mod)
        return mask


class FunctionPattern(Pattern):
    def __init__(self, func):
        self.func = func

    def create_mask(self, spectro_shape, spectro_mod, shape_factory):
        freqs = spectro_mod._get_freqs()
        times = spectro_mod._get_times()
        ff, tt = np.meshgrid(freqs, times, indexing='ij')
        return self.func(ff, tt)
