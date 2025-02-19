import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import logging
from tqdm import tqdm
import librosa
import librosa.display

from config import (
    NOISE_PIPE_LINE_DEBUG_LOGGING,
    MAX_LEVEL,
    TOTAL_FILES,
    OUTPUT_DIR,
    SR,
    DURATION,
    N_FFT,
    HOP_LENGTH,
    WINDOW,
    LOGGER_DIR,
)
from noise_pipeline.constants import (
    RATIO_SHAPE_BASE,
    RATIO_PATTERN_BASE,
    SHAPE_TYPE_MAPPING,
    PATTERN_TYPE_MAPPING,
)
from noise_pipeline import (
    SpectrogramModifier,
    NoisePipeline,
    create_random_noise_pipeline,
    reconstruct_audio_from_final_spectrogram,
    calculate_complexity_level,
)
from noise_pipeline.utils import pick_item_from_ratio
from noise_pipeline.shape_params import generate_shape_params


def setup_logging(debug=False):
    """
    로거(Logger)를 설정한다. debug=True일 경우 DEBUG 레벨로 설정,
    그렇지 않으면 WARNING 레벨로 설정한다.
    """
    log_level = logging.DEBUG if debug else logging.WARNING
    os.makedirs(LOGGER_DIR, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    fh = logging.FileHandler(os.path.join(LOGGER_DIR, "batch_level_main.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.getLogger("noise_pipeline.utils").setLevel(logging.WARNING)


def validate_params(total_files, sr, duration):
    """
    파라미터 유효성을 검증한다. 파일 개수, 샘플링 레이트(sr),
    신호 길이(duration)가 올바른 값인지 확인한다.
    """
    if total_files < 1 or sr <= 0 or duration <= 0:
        raise ValueError("Invalid parameters: total_files, sr, or duration.")


def distribute_files(total_files, max_level):
    """
    전체 파일(total_files)을 레벨 1부터 max_level까지 균등 분배한다.
    나누어떨어지지 않을 경우, 나머지(remainder)를 각 레벨에 하나씩
    추가해준다.
    """
    per_level = total_files // max_level
    remainder = total_files % max_level

    counts = {level: per_level for level in range(1, max_level + 1)}

    for level in range(1, max_level + 1):
        if remainder:
            counts[level] += 1
            remainder -= 1
    return counts


def gen_noise_params():
    """
    노이즈 타입(noise_type), 노이즈 세기(noise_strength), 노이즈 파라미터를
    무작위로 생성한다.
    """
    noise_types = ["normal", "uniform", "perlin"]
    noise_type = random.choice(noise_types)
    noise_strength = random.uniform(5, 15)
    seed = random.randint(0, int(1e9))

    if noise_type == "normal":
        params = {"mean": 0.0, "std": 1.0, "seed": seed}
    elif noise_type == "uniform":
        params = {"low": -1.0, "high": 1.0, "seed": seed}
    else:
        params = {"seed": seed, "scale": random.uniform(20.0, 100.0)}

    return noise_type, noise_strength, params


def init_spectro_mod(sr, strength, n_type, n_params, window, n_fft,
                     hop_length):
    """
    SpectrogramModifier 객체를 초기화한다.
    """
    return SpectrogramModifier(
        sample_rate=sr,
        noise_strength=strength,
        noise_type=n_type,
        noise_params=n_params,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
    )


def create_pipeline(spectro_mod, max_shapes, max_patterns, duration, sr,
                    ratio_shape, ratio_pattern):
    """
    랜덤한 노이즈 파이프라인(NoisePipeline)을 생성한다.
    shapes와 patterns의 최대 개수를 지정하고, 블러 적용 여부,
    블러 시그마, 기타 파라미터를 세팅한다.
    """
    return create_random_noise_pipeline(
        spectro_mod,
        max_shapes=max_shapes,
        max_patterns=max_patterns,
        apply_blur=True,
        blur_sigma=1.0,
        duration=duration,
        sr=sr,
        freq_min=0,
        min_float_value=0.001,
        alpha=1.0,
        ratio_shape=ratio_shape,
        ratio_pattern=ratio_pattern,
        max_db_power=40,
        min_db_power=20,
    )


def gen_spectrogram(pipeline, sr, duration):
    """
    지정된 파이프라인(pipeline)으로부터 스펙트로그램을 생성한다.
    """
    return pipeline.generate(int(sr * duration))


def save_fig(spectro_mod, show_labels, title, path):
    """
    SpectrogramModifier의 내부 스펙트로그램을 시각화하여
    파일로 저장한다.
    """
    fig, _ = spectro_mod.plot_spectrogram(
        show_labels=show_labels, title=title
    )
    fig.savefig(
        path,
        bbox_inches="tight" if not show_labels else None,
        pad_inches=0 if not show_labels else None,
        dpi=100,
    )
    plt.close(fig)


def save_audio(spectro_mod, path, sr):
    """
    최종 스펙트로그램에서 역변환하여 오디오를 생성하고, 파일로 저장한다.
    """
    audio = reconstruct_audio_from_final_spectrogram(spectro_mod)
    sf.write(path, audio, int(sr))


def save_metadata(json_path, audio_path, axes_path, no_axes_path,
                  spectro_mod, pipeline, level, duration):
    """
    JSON 메타데이터를 생성 및 저장한다.
    """
    meta = {
        "file_path": audio_path,
        "file_name": os.path.basename(audio_path),
        "spectrogram_with_axes": axes_path,
        "spectrogram_no_axes": no_axes_path,
        "spectrogram_base": {
            "sample_rate": spectro_mod.sample_rate,
            "n_fft": spectro_mod.n_fft,
            "hop_length": spectro_mod.hop_length,
        },
        "shapes": [
            {
                "type": SHAPE_TYPE_MAPPING.get(
                    shape.__class__.__name__, "unknown"
                ),
                "parameters": shape.__dict__,
            }
            for shape in pipeline.shapes
        ],
        "patterns": [
            {
                "type": PATTERN_TYPE_MAPPING.get(
                    pattern.__class__.__name__, "unknown"
                ),
                "parameters": pattern.__dict__,
            }
            for pattern in pipeline.patterns
        ],
        "complexity_level": level,
        "shape_count": len(
            {shape.__class__.__name__ for shape in pipeline.shapes}
        ),
        "pattern_count": len(pipeline.patterns),
        "duration": duration,
        "hz": spectro_mod.sample_rate / 2,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=4)


def clear_dir(path):
    """
    지정한 경로(path)에 파일이 존재하면 전부 삭제하고,
    디렉터리가 없으면 새로 생성한다.
    """
    if os.path.exists(path) and os.listdir(path):
        for root, _, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in os.listdir(root):
                os.rmdir(os.path.join(root, name))
    else:
        os.makedirs(path, exist_ok=True)


def create_dirs(base_dir):
    """
    결과물(오디오, 스펙트로그램, JSON 등)을 저장할 폴더를 생성한다.
    """
    paths = {
        "audio": os.path.join(base_dir, "audio"),
        "spectro_axes": os.path.join(
            base_dir, "linear_spectrogram_with_axes"
        ),
        "spectro_no_axes": os.path.join(
            base_dir, "linear_spectrogram_no_axes"
        ),
        "spectro_no_axes_from_audio": os.path.join(
            base_dir, "linear_spectrogram_no_axes_from_audio"
        ),
        "json": os.path.join(base_dir, "json"),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def save_spectrogram_from_audio(audio_path, sr, n_fft, hop_length,
                                window, show_labels, title, save_path):
    """
    이미 저장된 오디오 파일을 다시 불러와 STFT를 수행한 뒤,
    스펙트로그램을 시각화하여 저장한다.
    """
    audio, _ = librosa.load(audio_path, sr=sr)
    S = np.abs(
        librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window
        )
    )
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    fig, ax = plt.subplots(dpi=100)
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        ax=ax
    )

    if show_labels:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
    else:
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.tight_layout(pad=0.5)
    fig.savefig(
        save_path,
        bbox_inches="tight" if not show_labels else None,
        pad_inches=0 if not show_labels else None,
        dpi=100
    )
    plt.close(fig)


def batch_generate(max_level, total_files, out_dir, sr, duration,
                   n_fft, hop_length, window):
    """
    지정한 파라미터를 바탕으로 여러 개의 레벨(level)에 대해
    오디오와 스펙트로그램을 생성한다.
    """
    logger = logging.getLogger(__name__)

    validate_params(total_files, sr, duration)
    dirs = create_dirs(out_dir)
    counts = distribute_files(total_files, max_level)

    ratio_shape = {k: v or 0 for k, v in RATIO_SHAPE_BASE.items()}
    ratio_pattern = {k: v or 0 for k, v in RATIO_PATTERN_BASE.items()}

    file_idx = 1

    for level, count in counts.items():
        logger.info(f"Level {level}: {count} files")

        for _ in tqdm(range(count), desc=f"Level {level}"):
            fid = f"level_{level}_{file_idx:09d}"
            file_idx += 1

            n_type, n_strength, n_params = gen_noise_params()
            mod = init_spectro_mod(
                sr, n_strength, n_type, n_params,
                window, n_fft, hop_length
            )

            shapes = level
            patterns = max(0, level - 1)

            pipe = create_pipeline(
                mod, shapes, patterns, duration,
                sr, ratio_shape, ratio_pattern
            )

            gen_spectrogram(pipe, sr, duration)

            act_level = calculate_complexity_level(
                pipe.shapes, pipe.patterns
            )
            logger.info(f"File {fid}: level {act_level}")

            audio_path = os.path.join(
                dirs["audio"], f"audio_{fid}.wav"
            )
            axes_path = os.path.join(
                dirs["spectro_axes"], f"spectro_axes_{fid}.png"
            )
            no_axes_path = os.path.join(
                dirs["spectro_no_axes"], f"spectro_no_axes_{fid}.png"
            )
            no_axes_from_audio_path = os.path.join(
                dirs["spectro_no_axes_from_audio"],
                f"spectro_no_axes_from_audio_{fid}.png"
            )
            json_path = os.path.join(
                dirs["json"], f"audio_{fid}.json"
            )

            # 오디오 저장
            save_audio(mod, audio_path, sr)

            # 원본 스펙트로그램(축 표시/비표시) 저장
            save_fig(
                mod, True,
                f"Spectrogram Level {act_level}",
                axes_path
            )
            save_fig(
                mod, False,
                f"Spectrogram Level {act_level}",
                no_axes_path
            )

            # 저장된 오디오를 다시 불러와 만든 스펙트로그램
            save_spectrogram_from_audio(
                audio_path,
                sr,
                n_fft,
                hop_length,
                window,
                show_labels=False,
                title=f"Spectrogram Level {act_level}",
                save_path=no_axes_from_audio_path
            )

            # 메타데이터 저장
            save_metadata(
                json_path,
                audio_path,
                axes_path,
                no_axes_path,
                mod,
                pipe,
                act_level,
                duration
            )


def main():
    """
    메인 함수: 로거 설정, 출력 디렉터리 정리 후,
    batch_generate 함수를 호출한다.
    """
    setup_logging(NOISE_PIPE_LINE_DEBUG_LOGGING)
    clear_dir(OUTPUT_DIR)
    batch_generate(
        max_level=MAX_LEVEL,
        total_files=TOTAL_FILES,
        out_dir=OUTPUT_DIR,
        sr=SR,
        duration=DURATION,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=WINDOW,
    )


if __name__ == "__main__":
    main()
