# noise_pipeline/factories.py

from .shapes import (
    CircleDBShape,
    TrapezoidDBShape,
    RectangleDBShape,
    EllipseDBShape,
    FogDBShape,
    PolygonDBShape,
    WavePatternDBShape,
    RealWorldNoiseDBShape,
    HorizontalSpikeDBShape,
    VerticalSpikeDBShape,
    HillDBShape,
    PillarDBShape,
    HorizontalLineDBShape,
    VerticalLineDBShape,
    HorizontalRangeDistributionDBShape,
    VerticalRangeDistributionDBShape
)
from .patterns import (
    LinearPattern,
    RandomPattern,
    NLinearRepeatTSleepPattern,
    ConvexPattern,
    FunctionPattern
)

from .constants import SHAPE_TYPE_MAPPING, PATTERN_TYPE_MAPPING

class ShapeFactory:
    def create(self, shape_name, **kwargs):
        shape_name = shape_name.lower()
        if shape_name == "circle":
            return CircleDBShape(
                center_freq=kwargs['center_freq'],
                center_time=kwargs['center_time'],
                radius_freq=kwargs['radius_freq'],
                radius_time=kwargs['radius_time'],
                strength_dB=kwargs['strength_dB']
            )
        elif shape_name == "trapezoid":
            return TrapezoidDBShape(
                freq_min=kwargs['freq_min'],
                freq_max=kwargs['freq_max'],
                time_min=kwargs['time_min'],
                time_max=kwargs['time_max'],
                slope_freq=kwargs['slope_freq'],
                slope_time=kwargs['slope_time'],
                strength_dB=kwargs['strength_dB']
            )
        elif shape_name == "rectangle":
            return RectangleDBShape(
                freq_min=kwargs['freq_min'],
                freq_max=kwargs['freq_max'],
                time_min=kwargs['time_min'],
                time_max=kwargs['time_max'],
                strength_dB=kwargs['strength_dB']
            )
        elif shape_name == "ellipse":
            return EllipseDBShape(
                center_freq=kwargs['center_freq'],
                center_time=kwargs['center_time'],
                radius_freq=kwargs['radius_freq'],
                radius_time=kwargs['radius_time'],
                strength_dB=kwargs['strength_dB']
            )
        elif shape_name == "fog":
            return FogDBShape(
                strength_dB=kwargs['strength_dB'],
                coverage=kwargs.get('coverage', 1.0)
            )
        elif shape_name == "polygon":
            return PolygonDBShape(
                vertices=kwargs['vertices'],
                strength_dB=kwargs['strength_dB']
            )
        elif shape_name == "wave_pattern":
            return WavePatternDBShape(
                axis=kwargs.get('axis', 'time'),
                frequency=kwargs['frequency'],
                strength_dB=kwargs['strength_dB']
            )
        elif shape_name == "real_world_noise":
            return RealWorldNoiseDBShape(
                audio_path=kwargs['audio_path'],
                freq_min=kwargs['freq_min'],
                freq_max=kwargs['freq_max'],
                time_min=kwargs['time_min'],
                time_max=kwargs['time_max'],
                strength_dB=kwargs['strength_dB']
            )
        elif shape_name == "horizontal_spike":
            return HorizontalSpikeDBShape(
                center_freq=kwargs['center_freq'],
                center_time=kwargs['center_time'],
                radius_freq=kwargs['radius_freq'],
                radius_time=kwargs['radius_time'],
                strength_dB=kwargs['strength_dB']
            )
        elif shape_name == "vertical_spike":
            return VerticalSpikeDBShape(
                center_freq=kwargs['center_freq'],
                center_time=kwargs['center_time'],
                radius_freq=kwargs['radius_freq'],
                radius_time=kwargs['radius_time'],
                strength_dB=kwargs['strength_dB']
            )
        elif shape_name == "hill":
            return HillDBShape(
                freq_center=kwargs['freq_center'],
                time_center=kwargs['time_center'],
                freq_width=kwargs['freq_width'],
                time_width=kwargs['time_width'],
                strength_dB=kwargs['strength_dB']
            )
        elif shape_name == "pillar":
            return PillarDBShape(
                freq_min=kwargs['freq_min'],
                freq_max=kwargs['freq_max'],
                strength_dB=kwargs['strength_dB']
            )
        elif shape_name == "horizontal_line":
            return HorizontalLineDBShape(
                center_freq=kwargs['center_freq'],
                strength_dB=kwargs['strength_dB'],
                thickness=kwargs.get('thickness', 1)
            )
        elif shape_name == "vertical_line":
            return VerticalLineDBShape(
                center_time=kwargs['center_time'],
                strength_dB=kwargs['strength_dB'],
                thickness=kwargs.get('thickness', 1)
            )
        elif shape_name == "horizontal_range_dist_db":
            return HorizontalRangeDistributionDBShape(
                freq_min=kwargs['freq_min'],
                freq_max=kwargs['freq_max'],
                strength_dB=kwargs['strength_dB'],
                distribution=kwargs.get('distribution', 'gaussian'),
                distribution_params=kwargs.get('distribution_params', {})
            )
        elif shape_name == "vertical_range_dist_db":
            return VerticalRangeDistributionDBShape(
                time_min=kwargs['time_min'],
                time_max=kwargs['time_max'],
                strength_dB=kwargs['strength_dB'],
                distribution=kwargs.get('distribution', 'gaussian'),
                distribution_params=kwargs.get('distribution_params', {})
            )
        else:
            raise ValueError(f"Unknown shape name: {shape_name}")


class PatternFactory:
    def create(self, pattern_name, params):
        pattern_name = pattern_name.lower()
        if pattern_name == "linear":
            return LinearPattern(
                shape_name=params['shape_name'],
                shape_params=params['shape_params'],
                direction=params.get('direction', 'time'),
                repeat=params.get('repeat', 5),
                spacing=params.get('spacing', 1.0)
            )
        elif pattern_name == "random":
            return RandomPattern(
                shape_name=params['shape_name'],
                shape_params=params['shape_params'],
                n=params.get('n', 10),
                freq_range=params.get('freq_range', (0, 16000)),
                time_range=params.get('time_range', (0.0, 12.0))
            )
        elif pattern_name == "n_linear_repeat_t_sleep":
            direction = params.get('direction', 'time')
            if direction == 'freq':
                repeat_hz = params.get('repeat_time', 0.5) * 1000
            else:
                repeat_hz = None

            return NLinearRepeatTSleepPattern(
                shape_name=params['shape_name'],
                shape_params=params['shape_params'],
                repeat=params.get('repeat', 3),
                repeat_time=params.get('repeat_time', 0.5),
                repeat_hz=repeat_hz,
                sleep_time=params.get('sleep_time', 5.0),
                start_time=params.get('start_time', 0.0),
                direction=direction
            )
        elif pattern_name == "convex":
            return ConvexPattern(
                shape_name=params['shape_name'],
                shape_params=params['shape_params'],
                freq_min=params['freq_min'],
                freq_max=params['freq_max'],
                time_min=params['time_min'],
                time_max=params['time_max'],
                n=params.get('n', 10)
            )
        elif pattern_name == "function":
            return FunctionPattern(params['func'])
        else:
            return None
