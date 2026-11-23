"""Audio feature extraction: MFCC (librosa) or wav2vec 2.0 (transformers)."""

import numpy as np

SAMPLE_RATE = 16000
WAV2VEC2_MODEL = "facebook/wav2vec2-base-960h"
WAV2VEC2_RATE = 50.0  # output frames per second
MFCC_RATE = 100.0


def wav2vec2_features(audio_list, device="cpu", name=WAV2VEC2_MODEL):
    """Frozen wav2vec 2.0 hidden states ([T, 768] at 50 Hz) for a list of 16 kHz waveforms."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
    model = Wav2Vec2Model.from_pretrained(name).to(device).eval()
    out = []
    with torch.no_grad():
        for x in audio_list:
            values = extractor(x, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
            out.append(model(values.to(device)).last_hidden_state[0].float().cpu().numpy())
    return out, WAV2VEC2_RATE


def mfcc_features(audio_list):
    """40 MFCCs + deltas ([T, 80] at 100 Hz) for a list of 16 kHz waveforms."""
    import librosa

    out = []
    for x in audio_list:
        mfcc = librosa.feature.mfcc(y=x, sr=SAMPLE_RATE, n_mfcc=40, hop_length=160, n_fft=400)
        delta = librosa.feature.delta(mfcc)
        out.append(np.concatenate([mfcc, delta]).T.astype(np.float32))
    return out, MFCC_RATE


def extract_features(audio_list, kind, device="cpu"):
    if kind == "wav2vec2":
        return wav2vec2_features(audio_list, device)
    return mfcc_features(audio_list)


def resample_features(feat, rate, n_frames, fps):
    """Linearly interpolate a [T, C] feature sequence sampled at `rate` onto n_frames at fps."""
    t_src = np.arange(len(feat)) / rate
    t_dst = np.arange(n_frames) / fps
    idx = np.clip(np.searchsorted(t_src, t_dst) - 1, 0, len(feat) - 2)
    w = np.clip((t_dst - t_src[idx]) * rate, 0, 1)[:, None]
    return ((1 - w) * feat[idx] + w * feat[idx + 1]).astype(np.float32)

