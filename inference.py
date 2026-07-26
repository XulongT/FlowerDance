import argparse
import os
import pickle
import random
import sys
import types
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


FPS = 30
HOP_LENGTH = 512
SAMPLE_RATE = FPS * HOP_LENGTH
FEATURE_DIM = 35
MOTION_DIM = 151
DEFAULT_DURATION = 32.0
MAX_DURATION = 60.0

GENRES = (
    "Dai",
    "ShenYun",
    "Wei",
    "Korean",
    "Urban",
    "Hiphop",
    "Popping",
    "Miao",
    "HanTang",
    "Breaking",
    "Kun",
    "Locking",
    "Jazz",
    "Choreography",
    "Chinese",
    "DunHuang",
)

# These are the exact per-channel bounds used by the training-set condition
# normalizer. Keeping them here makes inference independent of the 3.4 GB
# cached training dataset.
COND_MIN = np.asarray(
    [
        0.0,
        -1131.3709716796875,
        -237.15911865234375,
        -171.30734252929688,
        -92.51953125,
        -113.9908447265625,
        -83.63716125488281,
        -91.29580688476562,
        -69.15321350097656,
        -77.8322525024414,
        -69.10548400878906,
        -87.16200256347656,
        -61.735443115234375,
        -80.8205795288086,
        -58.1674690246582,
        -80.16529083251953,
        -65.69412994384766,
        -69.32086181640625,
        -64.44036865234375,
        -68.75288391113281,
        -62.361083984375,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

COND_MAX = np.asarray(
    [
        40.581703186035156,
        155.65695190429688,
        289.90618896484375,
        153.26834106445312,
        148.48130798339844,
        106.52911376953125,
        104.7796401977539,
        81.25057983398438,
        83.61593627929688,
        83.17398071289062,
        120.94718933105469,
        74.80978393554688,
        90.64946746826172,
        65.08392333984375,
        74.385986328125,
        69.89390563964844,
        76.94829559326172,
        65.66349029541016,
        83.71824645996094,
        69.2318344116211,
        74.71946716308594,
        0.9658729434013367,
        0.9479968547821045,
        0.9964158535003662,
        0.9637431502342224,
        0.9991030693054199,
        0.9481787085533142,
        0.9748934507369995,
        1.0,
        0.9787861704826355,
        0.9942631721496582,
        0.9858856797218323,
        0.9759241342544556,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)


def parse_genre(value):
    value = str(value).strip()
    if value.isdigit():
        genre_id = int(value)
        if 0 <= genre_id < len(GENRES):
            return genre_id

    lowered = value.casefold()
    for genre_id, genre_name in enumerate(GENRES):
        if genre_name.casefold() == lowered:
            return genre_id

    valid = ", ".join(f"{index}:{name}" for index, name in enumerate(GENRES))
    raise argparse.ArgumentTypeError(f"Unknown genre '{value}'. Choose one of: {valid}")


def load_audio_clip(audio_path, start, duration):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    start_sample = round(start * SAMPLE_RATE)
    frame_count = round(duration * FPS)
    sample_count = frame_count * HOP_LENGTH
    end_sample = start_sample + sample_count

    if start_sample < 0:
        raise ValueError("--start must be non-negative")
    if end_sample > len(audio):
        available = max(0.0, len(audio) / SAMPLE_RATE - start)
        raise ValueError(
            f"Input audio is too short: requested {duration:.2f}s from "
            f"{start:.2f}s, but only {available:.2f}s is available."
        )

    tempo_audio, tempo_sample_rate = librosa.load(
        audio_path,
        sr=22050,
        mono=True,
        offset=start,
        duration=duration,
    )
    return (
        np.asarray(audio[start_sample:end_sample], dtype=np.float32),
        np.asarray(tempo_audio, dtype=np.float32),
        tempo_sample_rate,
        frame_count,
    )


def estimate_tempo(audio, sample_rate):
    tempo = librosa.beat.tempo(y=audio, sr=sample_rate)
    return float(np.asarray(tempo).reshape(-1)[0])


def extract_baseline_features(audio, frame_count, start_bpm=None):
    envelope = librosa.onset.onset_strength(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_mfcc=20,
    ).T
    chroma = librosa.feature.chroma_cens(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_chroma=12,
    ).T

    peak_indices = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_indices] = 1.0

    _, beat_indices = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        start_bpm=(
            estimate_tempo(audio, SAMPLE_RATE) if start_bpm is None else start_bpm
        ),
        tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[np.asarray(beat_indices, dtype=np.int64)] = 1.0

    common_length = min(
        len(envelope),
        len(mfcc),
        len(chroma),
        len(peak_onehot),
        len(beat_onehot),
    )
    if common_length < frame_count:
        raise RuntimeError(
            f"Feature extractor returned {common_length} frames, "
            f"but {frame_count} are required."
        )

    features = np.concatenate(
        [
            envelope[:frame_count, None],
            mfcc[:frame_count],
            chroma[:frame_count],
            peak_onehot[:frame_count, None],
            beat_onehot[:frame_count, None],
        ],
        axis=-1,
    )
    if features.shape != (frame_count, FEATURE_DIM):
        raise RuntimeError(f"Unexpected audio feature shape: {features.shape}")
    return features.astype(np.float32, copy=False)


def normalize_features(features):
    data_range = COND_MAX - COND_MIN
    if np.any(data_range <= 0):
        raise RuntimeError("Invalid embedded condition-normalization bounds")
    normalized = 2.0 * (features - COND_MIN) / data_range - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


def resolve_checkpoint(checkpoint):
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint.resolve()

    local_checkpoint = Path("runs/train/uniform2/weights/train-3700.pt")
    if local_checkpoint.is_file():
        return local_checkpoint.resolve()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "No local checkpoint was found and huggingface_hub is unavailable. "
            "Install the repository requirements or pass --checkpoint."
        ) from exc

    downloaded = hf_hub_download("xlt99/FlowerDance", "train-3700.pt")
    return Path(downloaded).resolve()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_edge():
    try:
        import p_tqdm  # noqa: F401
    except ImportError:
        compatibility_module = types.ModuleType("p_tqdm")
        compatibility_module.p_map = lambda function, values, **_: list(
            map(function, values)
        )
        sys.modules["p_tqdm"] = compatibility_module

    from EDGE import EDGE

    return EDGE


def load_model(checkpoint_path):
    if not torch.cuda.is_available():
        raise RuntimeError("FlowerDance inference requires a CUDA GPU.")

    EDGE = import_edge()
    model = EDGE(feature_type="baseline", checkpoint_path="")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=model.accelerator.device,
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint or "normalizer" not in checkpoint:
        raise KeyError(
            "Checkpoint must contain 'model_state_dict' and 'normalizer'."
        )

    unwrapped_model = model.accelerator.unwrap_model(model.model)
    unwrapped_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.normalizer = checkpoint["normalizer"]
    model.eval()
    return model


def generate_motion(model, features, genre_id, output_dir, output_stem, steps):
    device = model.accelerator.device
    condition = torch.from_numpy(features).unsqueeze(0).to(device)
    genre = torch.tensor([genre_id], dtype=torch.long, device=device)
    shape = (1, condition.shape[1], MOTION_DIM)

    with torch.inference_mode():
        model.flow_matching.render_sample(
            shape,
            condition,
            genre,
            model.normalizer,
            epoch=0,
            render_out=None,
            fk_out=str(output_dir),
            name=[f"{output_stem}.npy"],
            sound=False,
            n_steps=steps,
        )

    motion_path = output_dir / "0" / f"{output_stem}.pkl"
    if not motion_path.is_file():
        raise RuntimeError(f"Expected motion output was not created: {motion_path}")

    with motion_path.open("rb") as file:
        motion = pickle.load(file)
    expected_frames = features.shape[0]
    if motion["smpl_poses"].shape != (expected_frames, 72):
        raise RuntimeError(
            "Unexpected generated SMPL pose shape: "
            f"{motion['smpl_poses'].shape}"
        )
    return motion_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a FlowerDance motion from an uploaded music file."
    )
    parser.add_argument(
        "music",
        type=Path,
        nargs="?",
        help="Input WAV, MP3, FLAC, or OGG file",
    )
    parser.add_argument(
        "--genre",
        type=parse_genre,
        default=parse_genre("Hiphop"),
        help="Genre name or index. Default: Hiphop",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Downloads xlt99/FlowerDance when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inference_outputs"),
        help="Directory for generated motion and the processed audio clip.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Output duration in seconds. Default: {DEFAULT_DURATION:g}",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in the input music, in seconds.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=21,
        help="Number of Euler sampling steps. Default: 21",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--list-genres",
        action="store_true",
        help="Print supported genres and exit.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_genres:
        for genre_id, genre_name in enumerate(GENRES):
            print(f"{genre_id:2d}  {genre_name}")
        return

    if args.music is None:
        parser.error("music is required unless --list-genres is used")

    music_path = args.music.expanduser().resolve()
    if not music_path.is_file():
        parser.error(f"Music file not found: {music_path}")
    if not 0 < args.duration <= MAX_DURATION:
        parser.error(f"--duration must be in (0, {MAX_DURATION:g}] seconds")
    if args.steps < 2:
        parser.error("--steps must be at least 2")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{music_path.stem}_{GENRES[args.genre]}"

    print(f"Music: {music_path}")
    print(f"Genre: {GENRES[args.genre]} ({args.genre})")
    print(f"Duration: {args.duration:.2f}s from {args.start:.2f}s")

    audio, tempo_audio, tempo_sample_rate, frame_count = load_audio_clip(
        str(music_path),
        start=args.start,
        duration=args.duration,
    )
    start_bpm = estimate_tempo(tempo_audio, tempo_sample_rate)
    raw_features = extract_baseline_features(audio, frame_count, start_bpm)
    normalized_features = normalize_features(raw_features)

    audio_output = output_dir / f"{output_stem}.wav"
    feature_output = output_dir / f"{output_stem}_features.npy"
    sf.write(audio_output, audio, SAMPLE_RATE)
    np.save(feature_output, normalized_features)

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"Checkpoint: {checkpoint_path}")
    set_seed(args.seed)
    model = load_model(checkpoint_path)
    motion_path = generate_motion(
        model,
        normalized_features,
        args.genre,
        output_dir,
        output_stem,
        args.steps,
    )

    print(f"Motion: {motion_path}")
    print(f"Audio clip: {audio_output}")
    print(f"Normalized features: {feature_output}")


if __name__ == "__main__":
    # Avoid tokenizer worker processes being created by transitive imports.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
