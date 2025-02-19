# noise_pipeline/shape_params.py

import random
import logging
from .utils import clamp  # Ensure clamp is accessible, adjust as necessary

logger = logging.getLogger(__name__)

def generate_shape_params(
    shape_name,
    duration,
    sr,
    freq_min=20,
    time_min=0.0,
    min_float_value=0.001,
    alpha=1.0,
    max_db_power=20,
    min_db_power=10
):
    """
    Generate parameters for a specific shape, ensuring values are within specified boundaries.

    Args:
        shape_name (str): Name of the shape.
        duration (float): Duration of the audio in seconds.
        sr (int): Sample rate.
        freq_min (float, optional): Minimum frequency. Defaults to 20.
        time_min (float, optional): Minimum time. Defaults to 0.0.
        min_float_value (float, optional): Minimum float value for certain parameters. Defaults to 0.001.
        alpha (float, optional): Scaling factor. Defaults to 1.0.
        max_db_power (float, optional): Maximum dB power for shapes. Defaults to 20.
        min_db_power (float, optional): Minimum dB power for shapes. Defaults to 10.

    Returns:
        dict: Generated parameters for the shape.
    """
    logger.debug(f"Generating shape parameters for shape: {shape_name}")

    params = {
        "strength_dB": random.uniform(min_db_power, max_db_power)
    }

    # Generate frequency bounds
    random_freq_min = random.uniform(freq_min, sr)
    random_freq_max = random.uniform(freq_min, sr)
    if random_freq_min > random_freq_max:
        random_freq_min, random_freq_max = random_freq_max, random_freq_min

    # Generate time bounds
    random_time_min = random.uniform(time_min, duration)
    random_time_max = random.uniform(time_min, duration)
    if random_time_min > random_time_max:
        random_time_min, random_time_max = random_time_max, random_time_min

    # Shape-specific parameter generation
    if shape_name == "horizontal_spike":
        params.update({
            "center_freq": clamp(
                random.uniform(random_freq_min, random_freq_max),
                0,
                sr
            ),
            "center_time": clamp(
                random.uniform(random_time_min, random_time_max),
                0,
                duration
            ),
            "radius_freq": clamp(
                random.uniform(30.0, 100.0),
                15.0,
                sr / 2
            ),
            "radius_time": clamp(
                random.uniform(random_time_max * 0.001, random_time_max * 0.1),
                0.05,
                duration * alpha
            )
        })
    elif shape_name == "vertical_spike":
        params.update({
            "center_freq": clamp(
                random.uniform(random_freq_min, random_freq_max) * 0.001,
                0,
                sr
            ),
            "center_time": clamp(
                random.uniform(random_time_min, random_time_max),
                0,
                duration
            ),
            "radius_freq": clamp(
                random.uniform(random_freq_max * 0.1, random_freq_max),
                0.1 * sr,
                sr
            ),
            "radius_time": clamp(
                random.uniform(random_time_max * 0.01, random_time_max * 0.1),
                0.05,
                duration * alpha
            )
        })
    elif shape_name == "trapezoid":
        params.update({
            "freq_min": clamp(random_freq_min, freq_min, sr),
            "freq_max": clamp(random_freq_max, freq_min, sr),
            "time_min": clamp(random_time_min, time_min, duration),
            "time_max": clamp(random_time_max, time_min, duration),
            "slope_freq": random.uniform(0.0, 2.0),
            "slope_time": random.uniform(0.0, 2.0)
        })
    elif shape_name == "fog":
        params.update({
            "coverage": clamp(random.uniform(0, 1), 0, 1)
        })
    elif shape_name == "hill":
        freq_center = random.uniform(random_freq_min, random_freq_max)
        center_time = random.uniform(random_time_min, random_time_max)
        freq_width = random.uniform(min_float_value, (random_freq_max - random_freq_min) * alpha)
        time_width = random.uniform(min_float_value, duration * alpha)

        params.update({
            "freq_center": clamp(freq_center, 0, sr),
            "time_center": clamp(center_time, 0, duration),
            "freq_width": clamp(freq_width, min_float_value, sr),
            "time_width": clamp(time_width, min_float_value, duration)
        })
    elif shape_name == "wave_pattern":
        params.update({
            "axis": random.choice(["time", "freq"]),
            "frequency": clamp(random.uniform(0.1, 10), 0.1, 10)
        })
    elif shape_name == "polygon":
        num_vertices = random.randint(3, 8)
        vertices = []
        for _ in range(num_vertices):
            f_rand = random.uniform(random_freq_min, random_freq_max)
            t_rand = random.uniform(random_time_min, random_time_max)
            f_rand = clamp(f_rand, 0, sr)
            t_rand = clamp(t_rand, 0, duration)
            vertices.append([f_rand, t_rand])
        params.update({
            "vertices": vertices
        })
    elif shape_name == "rectangle":
        params.update({
            "freq_min": clamp(random_freq_min, freq_min, sr),
            "freq_max": clamp(random_freq_max, freq_min, sr),
            "time_min": clamp(random_time_min, time_min, duration),
            "time_max": clamp(random_time_max, time_min, duration)
        })
    elif shape_name == "pillar":
        params.update({
            "freq_min": clamp(random_freq_min, freq_min, sr),
            "freq_max": clamp(random_freq_max, freq_min, sr)
        })
    elif shape_name == "horizontal_line":
        center_freq = random.uniform(random_freq_min, random_freq_max)
        params.update({
            "center_freq": clamp(center_freq, 0, sr),
            "thickness": random.randint(1, 3)
        })
    elif shape_name == "vertical_line":
        center_time = random.uniform(random_time_min, random_time_max)
        params.update({
            "center_time": clamp(center_time, 0, duration),
            "thickness": random.randint(1, 3)
        })
    elif shape_name == "horizontal_range_dist_db":
        chosen_dist = "gaussian"
        freq_min_range = clamp(
            random_freq_min - 50,
            0,
            sr
        )
        freq_max_range = clamp(
            random_freq_min + 50,
            0,
            sr
        )
        sigma = clamp(
            random.uniform(1000.0 * alpha, 1100.0 * alpha),
            1000.0 * alpha,
            1100.0 * alpha
        )

        params.update({
            "freq_min": freq_min_range,
            "freq_max": freq_max_range,
            "distribution": chosen_dist,
            "distribution_params": {"sigma": sigma}
        })
    elif shape_name == "vertical_range_dist_db":
        chosen_dist = "gaussian"
        time_min_range = clamp(
            random_time_min - 0.1,
            0,
            duration
        )
        time_max_range = clamp(
            random_time_max + 0.1,
            0,
            duration
        )
        sigma = clamp(
            random.uniform(1.0 * alpha, 2.0 * alpha),
            1.0 * alpha,
            2.0 * alpha
        )

        params.update({
            "time_min": time_min_range,
            "time_max": time_max_range,
            "distribution": chosen_dist,
            "distribution_params": {"sigma": sigma}
        })
    elif shape_name == "ellipse":
        center_freq = random.uniform(random_freq_min, random_freq_max)
        center_time = random.uniform(random_time_min, random_time_max)
        radius_freq = random.uniform(10, (random_freq_max - random_freq_min) * alpha)
        radius_time = random.uniform(min_float_value, duration * alpha)

        params.update({
            "center_freq": clamp(center_freq, 0, sr),
            "center_time": clamp(center_time, 0, duration),
            "radius_freq": clamp(radius_freq, 10, sr),
            "radius_time": clamp(radius_time, min_float_value, duration)
        })
    else:
        # Default handling for shapes like circle
        center_freq = random.uniform(random_freq_min, random_freq_max)
        center_time = random.uniform(random_time_min, random_time_max)
        radius_freq = random.uniform(10, (random_freq_max - random_freq_min) * alpha)
        radius_time = random.uniform(min_float_value, duration * alpha)

        params.update({
            "center_freq": clamp(center_freq, 0, sr),
            "center_time": clamp(center_time, 0, duration),
            "radius_freq": clamp(radius_freq, 10, sr),
            "radius_time": clamp(radius_time, min_float_value, duration)
        })

    logger.debug(f"Generated parameters for shape '{shape_name}': {params}")

    return params
