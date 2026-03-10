

from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

# --------------------------- CONFIG ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
LIBRISPEECH_FLAC_ROOT = PROJECT_ROOT / "dataset" / "dev-clean"
OUT_ROOT = PROJECT_ROOT / "dataset" / "dev-clean-raw20"

TARGET_SR = 16_000

FRAME_SEC = 0.020   # durata di ogni frame in secondi
HOP_SEC = 0.020     # passo tra frame (metti 0.10 per 50% overlap, ecc.)

FRAME_SAMPLES = int(FRAME_SEC * TARGET_SR)  # 0.20 * 16000 = 3200
HOP_SAMPLES = int(HOP_SEC * TARGET_SR)


# ------------------------ AUDIO UTILS -------------------------------

def load_waveform_mono(path: Path, target_sr: int):
    wav, sr = torchaudio.load(str(path))

    # mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample se necessario
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr

    return wav.squeeze(0), sr   # [samples]


def wav_to_frames(signal: torch.Tensor) -> torch.Tensor:
    """
    signal: Tensor [num_samples]
    ritorna: Tensor [num_frames, FRAME_SAMPLES]
    """

    num_samples = signal.numel()

    if num_samples < FRAME_SAMPLES:
        # opzionale: pad per avere almeno un frame
        pad_len = FRAME_SAMPLES - num_samples
        signal = torch.nn.functional.pad(signal, (0, pad_len))
        num_samples = signal.numel()

    # usa unfold per creare finestre 1D
    num_frames = 1 + (num_samples - FRAME_SAMPLES) // HOP_SAMPLES
    if num_frames <= 0:
        return torch.empty(0, FRAME_SAMPLES)

    frames = signal.unfold(0, FRAME_SAMPLES, HOP_SAMPLES)  # [num_frames, FRAME_SAMPLES]
    return frames.contiguous()


# ------------------------------ MAIN --------------------------------

def main():
    if not LIBRISPEECH_FLAC_ROOT.exists():
        raise FileNotFoundError(f"Cartella non trovata: {LIBRISPEECH_FLAC_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    flac_paths = sorted(LIBRISPEECH_FLAC_ROOT.rglob("*.flac"))
    print(f"Trovati {len(flac_paths)} file .flac sotto {LIBRISPEECH_FLAC_ROOT}")
    print(f"FRAME_SAMPLES = {FRAME_SAMPLES}, HOP_SAMPLES = {HOP_SAMPLES}")

    for i, flac_path in enumerate(flac_paths, start=1):
        rel = flac_path.relative_to(LIBRISPEECH_FLAC_ROOT)
        out_path = OUT_ROOT / rel.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            print(f"[{i}/{len(flac_paths)}] SKIP (esiste): {out_path}")
            continue

        waveform, sr = load_waveform_mono(flac_path, TARGET_SR)
        frames = wav_to_frames(waveform)  # [num_frames, FRAME_SAMPLES]

        torch.save(frames, out_path)

        print(
            f"[{i}/{len(flac_paths)}] {flac_path.name} -> {out_path} "
            f"shape={tuple(frames.shape)}"
        )

    print("Finito preprocessing raw 0.020 s.")


if __name__ == "__main__":
    main()
