# noise_pipeline/noise_pipeline.py

import numpy as np
from scipy.ndimage import gaussian_filter
# noise_pipeline/noise_pipeline.py (continued)

from .factories import ShapeFactory, PatternFactory


class NoisePipeline:
    def __init__(
        self,
        spectro_mod,
        apply_blur=False,
        blur_sigma=1.0
    ):
        self.spectro_mod = spectro_mod
        self.shape_factory = ShapeFactory()
        self.pattern_factory = PatternFactory()
        self.shapes = []
        self.patterns = []
        self.apply_blur = apply_blur
        self.blur_sigma = blur_sigma

    def add_shape(self, shape):
        self.shapes.append(shape)
        return self

    def add_pattern(self, pattern):
        if pattern is not None:
            self.patterns.append(pattern)
        return self

    def generate(self, length):
        base_signal = np.zeros(length)
        spec = self.spectro_mod.compute_spectrogram(base_signal)
        total_mask = np.zeros_like(spec)

        for shape in self.shapes:
            shape_mask = shape.create_mask(spec.shape, self.spectro_mod)
            total_mask += shape_mask

        for pattern in self.patterns:
            pattern_mask = pattern.create_mask(
                spec.shape,
                self.spectro_mod,
                self.shape_factory
            )
            total_mask += pattern_mask

        if self.apply_blur and self.blur_sigma > 0:
            total_mask = gaussian_filter(total_mask, sigma=self.blur_sigma)

        self.spectro_mod.apply_dB_mask(total_mask)
        return self.spectro_mod.S_db

