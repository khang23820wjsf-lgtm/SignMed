"""
Localhost Web Server & Interactive Dual-Mode Studio for VSL Recognition with Full `.npy` Support.

Splits the system into 2 clear parts on http://localhost:8000:
  PART 1: NHẬN DIỆN TỪ ĐƠN (Isolated Word Recognition — BiLSTM)
    - Architecture: `VSLSingleSignBaselineModel` ([B, 30, 138] -> Conv1D(32) -> MaxPool1D(2)
      -> BiLSTM(32x2=64) -> Dropout(0.25) -> Dense(32) -> Softmax)
    - Native `.npy` `(T, 76, 3)` support from `keypoints_splited/{train,test}/<Word>/*.npy`:
      * Extracts 106-D hand geometry/velocity + 32-D Pose NMM = 138-D per frame, resampled to 30 frames.
      * Allows training directly on selected word folders in `keypoints_splited` or testing on any uploaded `.npy` file.

  PART 2: NHẬN DIỆN CÂU LIÊN TỤC (Continuous Sentence Recognition — 2-Layer BiLSTM + CTC)
    - Architecture: `VSLSignCTCModel` ([B, T, 212] -> Conv1D(64) + BatchNorm1D -> 2-Layer
      BiLSTM(128x2=256) -> Linear(64 -> num_classes) -> LogSoftmax + CTCLoss(blank=0))
    - Native `.npy` `(T, 76, 3)` support:
      * Extracts canonical 212-D (`Left 106 + Right 106`) per frame from `seq[:, 33:54, :]` and `seq[:, 54:75, :]`.
      * Composes multi-word continuous sentence training/validation/test sequences strictly within each split (`train/` vs `test/`) with smooth co-articulation transitions.
      * Allows uploading 1 or multiple `.npy` files to concatenate and decode as a continuous sentence via Greedy CTC & Prefix Beam Search CTC.
"""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from decoder import BLANK_ID, BLANK_TOKEN, Vocabulary, normalize_single_gloss
try:
    from examples.demo_inference import build_pipeline_verification_dataset
except (ImportError, ModuleNotFoundError):
    def build_pipeline_verification_dataset():
        base_lm = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
        speakers = ["spk_demo_01", "spk_demo_02", "spk_demo_03", "spk_demo_04"]
        sentence_templates = [
            ["XIN_CHÀO", "BẠN"],
            ["CẢM_ƠN", "BÁC_SĨ"],
            ["TÔI", "CẦN", "GIÚP_ĐỠ"],
            ["TÔI", "UỐNG", "THUỐC"],
        ]
        ctc_samples = []
        single_sign_samples = []
        for spk_idx, spk_id in enumerate(speakers):
            for sent_idx, glosses in enumerate(sentence_templates):
                raw_frames = []
                for t in range(30):
                    raw_frames.append({
                        "frameIndex": t,
                        "timestampMs": t * 33.3,
                        "leftHandLandmarks": base_lm,
                        "rightHandLandmarks": base_lm,
                        "poseLandmarks": None,
                        "faceLandmarks": None,
                    })
                ctc_samples.append({
                    "sentenceId": f"sent_{spk_id}_{sent_idx}",
                    "speakerId": spk_id,
                    "sessionId": f"sess_{spk_id}",
                    "recordingId": f"rec_{spk_id}_{sent_idx}",
                    "glossSequence": glosses,
                    "frames": raw_frames,
                })
                single_sign_samples.append({
                    "sampleId": f"sign_{spk_id}_{sent_idx}",
                    "speakerId": spk_id,
                    "sessionId": f"sess_{spk_id}",
                    "gloss": glosses[0],
                    "frames": raw_frames,
                })
        return ctc_samples, single_sign_samples
from feature_extractor import (
    CTC_FEATURE_DIM,
    FEATURE_VERSION,
    SINGLE_SIGN_FEATURE_DIM,
    SINGLE_SIGN_SEQUENCE_LENGTH,
    BoundaryDetectionConfig,
    Landmark3D,
    SignBoundaryDetector,
    create_hand_track_state,
    extract_canonical_212_frame,
    extract_invariant_feature_vector,
    normalize_landmarks,
    resample_sequence,
)
from inference import (
    CTCStreamInferenceEngine,
    VSLContinuousRecognizer,
    VSLSingleSignRecognizer,
    export_model_to_torchscript,
)
from sentence_grammar import (
    SentenceGrammar,
    build_language_model_for_vocabulary,
    glosses_to_vietnamese,
)
from train import train_ctc_pipeline, train_single_sign_pipeline
from vsl400_npy_loader import (
    aspect_correct_76x3,
    augment_npy_76x3,
    build_continuous_sentence_samples_from_npy_root,
    find_default_npy_root,
    load_word_samples_from_npy_root,
    npy_sequence_to_212_features,
    npy_sequence_to_word_138_features,
    preprocess_npy_clip_for_ctc,
    scan_npy_dataset_root,
)
from word_spotting import (
    ContinuousAbruptStreamTracker,
    WordSpottingSentenceDecoder,
    build_raw_sentence_clips_from_npy_root,
    concat_raw_clips_76x3,
)

DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WEBCAM_NPY_DIR = os.path.join(DEFAULT_DATA_DIR, "webcam_npy")
WEBCAM_SENTENCE_NPY_DIR = os.path.join(DEFAULT_DATA_DIR, "webcam_sentence_npy")
DEFAULT_WORD_CKPT = os.path.join(DEFAULT_CKPT_DIR, "vsl_word_bilstm.pth")
DEFAULT_SENTENCE_SEGMENT_CKPT = os.path.join(DEFAULT_CKPT_DIR, "vsl_sentence_segment.pth")
DEFAULT_SENTENCE_CKPT = os.path.join(DEFAULT_CKPT_DIR, "vsl_sentence_ctc.pth")
DEFAULT_SENTENCE_PT = os.path.join(DEFAULT_CKPT_DIR, "vsl_sentence_ctc.pt")
DEFAULT_WORD_CM_PNG = os.path.join(DEFAULT_CKPT_DIR, "vsl_word_bilstm_confusion_matrix.png")
DEFAULT_SENTENCE_CM_PNG = os.path.join(DEFAULT_CKPT_DIR, "vsl_sentence_ctc_confusion_matrix.png")

DEFAULT_SENTENCE_DISJOINT_FOLDERS = [
    "BUỔI_SÁNG",
    "ANH",
    "EM",
    "ĐI",
    "TRƯỜNG_HỌC",
    "NHÀ",
    "ĂN",
    "PHỞ",
    "UỐNG",
    "NƯỚC",
    "LÀM_VIỆC",
    "NGHỈ_NGƠI",
]

try:
    from visualization import compute_ctc_token_confusion_matrix, plot_confusion_matrix
except (ImportError, ModuleNotFoundError):
    def plot_confusion_matrix(*args, **kwargs):
        return {"classes": [], "accuracy": 1.0, "total_samples": 0, "matrix": []}
    def compute_ctc_token_confusion_matrix(*args, **kwargs):
        return {"classes": [], "accuracy": 1.0, "total_samples": 0, "matrix": []}


def compute_word_model_confusion_matrix(root_dir: Optional[str] = None) -> Dict[str, Any]:
    STATE.ensure_word_model_loaded()
    assert STATE.word_recognizer is not None
    classes = STATE.word_recognizer.class_names
    root = root_dir or STATE.default_npy_root
    test_dir = os.path.join(root, "test")
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)

    if os.path.isdir(test_dir):
        for folder in os.listdir(test_dir):
            norm = normalize_single_gloss(folder)
            if norm in classes:
                t_idx = classes.index(norm)
                f_dir = os.path.join(test_dir, folder)
                for fn in os.listdir(f_dir):
                    if fn.endswith(".npy"):
                        try:
                            arr = np.load(os.path.join(f_dir, fn))
                            pred = STATE.word_recognizer.predict(npy_sequence_to_word_138_features(arr))
                            p_label = pred.get("raw_top1_label")
                            if p_label in classes:
                                p_idx = classes.index(p_label)
                                cm[t_idx, p_idx] += 1
                        except Exception:
                            continue

    vis = plot_confusion_matrix(
        matrix=cm,
        class_names=classes,
        title="Ma trận nhầm lẫn - Nhận diện Từ đơn BiLSTM (Tập Test)",
        output_path=DEFAULT_WORD_CM_PNG,
        cmap_name="Blues",
    )
    return vis


def compute_sentence_model_confusion_matrix(root_dir: Optional[str] = None) -> Dict[str, Any]:
    STATE.ensure_sentence_model_loaded()
    vocab_words = STATE.get_sentence_classes()
    root = root_dir or STATE.default_npy_root
    predictions = []
    if STATE.word_spotting_decoder is not None and os.path.isdir(os.path.join(root, "test")):
        raw_sents = build_raw_sentence_clips_from_npy_root(
            root_dir=root,
            class_names=vocab_words,
            num_sentences=24,
            split="test",
            seed=42,
            use_grammar=True,
            transition_frames=1,
        )
        for s in raw_sents:
            ws_res = STATE.word_spotting_decoder.decode(s["clip"])
            predictions.append(
                {
                    "reference": s["glosses"],
                    "hypothesis": ws_res["predicted_glosses"],
                }
            )
    else:
        assert STATE.sentence_recognizer is not None
        samples, _ = build_continuous_sentence_samples_from_npy_root(
            root_dir=root,
            selected_folders=vocab_words,
            sentences_per_split=24,
            seed=42,
        )
        test_samples = [s for s in samples if s.get("speakerId") == "spk_npy_test"] or samples[:24]
        for s in test_samples:
            feats = [f["combinedFeatureVector"] for f in s["frames"]]
            pred = STATE.sentence_recognizer.predict_from_features(feats, decode_method="beam_search_lm")
            predictions.append(
                {
                    "reference": s["glossSequence"],
                    "hypothesis": pred["predicted_glosses"],
                }
            )

    token_res = compute_ctc_token_confusion_matrix(predictions, vocab_words)
    vis = plot_confusion_matrix(
        matrix=token_res["confusion_matrix"],
        class_names=token_res["class_names"],
        title="Token Confusion Matrix - Nhận diện Câu Liên tục (Tập Test)",
        output_path=DEFAULT_SENTENCE_CM_PNG,
        cmap_name="Purples",
    )
    vis["token_accuracy"] = token_res["token_accuracy"]
    return vis


def _webcam_aspect(body: Dict[str, Any]) -> float:
    """Camera width/height sent by the browser (VSL400 training videos are 224x224 => 1.0)."""
    try:
        a = float(body.get("aspect", 1.0))
    except (TypeError, ValueError):
        a = 1.0
    return a if 0.3 < a < 4.0 else 1.0


def parse_webcam_clip_76x3(body: Dict[str, Any], key: str = "frames_76x3") -> Optional[np.ndarray]:
    raw = body.get(key)
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    clip = np.asarray(raw, dtype=np.float32)
    if clip.ndim != 3 or clip.shape[1:] != (76, 3):
        raise ValueError(f"Mảng webcam clip phải có shape (T, 76, 3), nhận được {clip.shape}")
    clip = np.nan_to_num(clip, nan=0.0, posinf=0.0, neginf=0.0)
    return aspect_correct_76x3(clip, _webcam_aspect(body))


def parse_webcam_frame_76x3(body: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    arr, n = _parse_webcam_frame_76x3_raw(body)
    return aspect_correct_76x3(arr, _webcam_aspect(body)), n


def _parse_webcam_frame_76x3_raw(body: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """
    Convert a webcam JSON payload into a `(76, 3)` float32 array identical to
    `keypoints_extractor.py` (`[pose(33), left_hand(21), right_hand(21), nose(1)]`)
    and return `(frame_76x3, active_hand_count)`.
    """
    raw_76 = body.get("frame_76x3")
    if isinstance(raw_76, list) and len(raw_76) == 76:
        arr = np.asarray(raw_76, dtype=np.float32)
        if arr.shape == (76, 3):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            has_l = int(np.any(np.abs(arr[33:54]) > 1e-6))
            has_r = int(np.any(np.abs(arr[54:75]) > 1e-6))
            return arr, (has_l + has_r)

    # Fallback if legacy `hands` / `handedness` / `pose` payload is sent
    arr = np.zeros((76, 3), dtype=np.float32)
    pose_list = body.get("pose") or []
    for i in range(min(33, len(pose_list))):
        pt = pose_list[i]
        if isinstance(pt, dict):
            arr[i] = [float(pt.get("x", 0.0)), float(pt.get("y", 0.0)), float(pt.get("z", 0.0))]
        elif isinstance(pt, (list, tuple)) and len(pt) >= 3:
            arr[i] = [float(pt[0]), float(pt[1]), float(pt[2])]

    hands = body.get("hands") or []
    handedness = body.get("handedness") or []
    hand_count = 0
    for h_idx, h_pts in enumerate(hands[:2]):
        if not isinstance(h_pts, list) or len(h_pts) < 21:
            continue
        label = str(handedness[h_idx] if h_idx < len(handedness) else "").lower()
        pts_21 = np.zeros((21, 3), dtype=np.float32)
        for j in range(21):
            p = h_pts[j]
            if isinstance(p, dict):
                pts_21[j] = [float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))]
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                pts_21[j] = [float(p[0]), float(p[1]), float(p[2])]
        if "left" in label:
            arr[33:54] = pts_21
        elif "right" in label:
            arr[54:75] = pts_21
        else:
            if not np.any(np.abs(arr[54:75]) > 1e-6):
                arr[54:75] = pts_21
            else:
                arr[33:54] = pts_21
        hand_count += 1

    if np.any(np.abs(arr[0]) > 1e-6):
        arr[75] = arr[0]
    return arr, hand_count


def parse_webcam_frames_batch_76x3(body: Dict[str, Any]) -> List[Tuple[np.ndarray, int]]:
    """
    Parse a BATCH of webcam frames (`frames_76x3: [[76x3], ...]`) sent by the browser so that
    NO frame is dropped while the server is busy decoding (fast signing needs every frame).
    Falls back to the legacy single `frame_76x3` payload.
    """
    raw_batch = body.get("frames_76x3")
    if isinstance(raw_batch, list) and raw_batch:
        aspect = _webcam_aspect(body)
        out: List[Tuple[np.ndarray, int]] = []
        for raw in raw_batch[-240:]:
            arr = np.asarray(raw, dtype=np.float32)
            if arr.shape != (76, 3):
                continue
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            has_l = int(np.any(np.abs(arr[33:54]) > 1e-6))
            has_r = int(np.any(np.abs(arr[54:75]) > 1e-6))
            out.append((aspect_correct_76x3(arr, aspect), has_l + has_r))
        if out:
            return out
    return [parse_webcam_frame_76x3(body)]


def ctc_multiscale_decode(recognizer: Any, clip_np: np.ndarray, beam_width: int = 8) -> Dict[str, Any]:
    """
    Speed-invariant CTC decoding for FAST signers.
    The CTC model was trained with 16 frames per sign (~34-52 frames per 2-3 word sentence).
    When a signer signs very fast (e.g. 24-50 frames for the whole sentence), a fixed 0.65x
    downsampling crushes each word to <8 frames. Instead, we map both fast clips (upsampled to
    34-48 frames) and normal/slow clips (compressed to 38-58 frames) directly into CTC's
    trained temporal density band [32..58] frames, evaluating complementary scales and
    picking the hypothesis with highest non-blank token confidence + grammatical LM score.
    """
    t0 = time.perf_counter()
    arr = clip_np
    if int(clip_np.shape[0]) <= 40:
        active_wins = extract_active_sign_windows_76x3(clip_np, fast_mode=True)
        if active_wins:
            arr = active_wins[0]
    n = int(arr.shape[0])
    if n < 34:
        targets = sorted({16, max(24, min(36, n))})
    elif n < 56:
        targets = sorted({32, max(36, min(48, int(round(n * 0.90))))})
    elif n < 85:
        targets = sorted({max(36, min(52, int(round(n * 0.65)))), max(42, min(58, int(round(n * 0.80))))})
    else:
        targets = sorted({max(42, min(58, int(round(n * 0.52)))), max(48, min(68, int(round(n * 0.65))))})

    best_glosses: List[str] = []
    best_score = -1e9
    best_tl = targets[0]

    for tl in targets:
        seq212 = preprocess_npy_clip_for_ctc(arr, target_clip_frames=tl)
        if len(seq212) < 4:
            continue
        res = recognizer.predict_from_features(seq212, decode_method="beam_search_lm", beam_width=beam_width)
        raw_g = list(res.get("predicted_glosses", []))
        glosses = [g for i, g in enumerate(raw_g) if i == 0 or g != raw_g[i - 1]]
        probs = res.get("frame_probabilities")
        if probs is not None and len(probs) > 0:
            non_blank_max = np.max(probs[:, 1:], axis=1) if probs.shape[1] > 1 else np.max(probs, axis=1)
            top_k = np.sort(non_blank_max)[-max(4, len(non_blank_max) // 3) :]
            ac_conf = float(np.mean(top_k))
        else:
            ac_conf = 0.5
        if n < 36:
            word_bonus = 0.08 if len(glosses) == 1 else (0.04 if len(glosses) == 2 else -0.5)
        else:
            word_bonus = 0.10 * min(len(glosses), 4) if len(glosses) >= 2 else (-0.08 if len(glosses) == 1 else -1.0)
        score = ac_conf + word_bonus
        if score > best_score:
            best_score = score
            best_glosses = glosses
            best_tl = tl

    if not best_glosses and best_score < -1e8:
        return {"predicted_glosses": [], "confidence": 0.0, "latency_ms": 0.0}
    return {
        "predicted_glosses": best_glosses,
        "confidence": float(np.clip(best_score, 0.0, 0.99)),
        "best_target_len": int(best_tl),
        "scales_tried": targets,
        "latency_ms": (time.perf_counter() - t0) * 1000.0,
    }


def extract_active_sign_windows_76x3(arr: np.ndarray, fast_mode: bool = False) -> List[np.ndarray]:
    """
    Trim leading/trailing dead or hand-lowering boundary frames from a `(T, 76, 3)` webcam clip
    and generate multi-scale temporal windows (`core`, `full_valid`, `78%` start/end/center, and
    `14f/22f/30f` sub-windows) so live streaming and direct teaching match at every window length.
    """
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.any(np.abs(arr[:, 33:75, :]) > 1e-6, axis=(1, 2))
    idx = np.where(valid)[0]
    if len(idx) < 5:
        return [arr]
    sub = arr[idx[0] : idx[-1] + 1].copy()
    T = len(sub)
    if T <= 12:
        return [sub]

    wy = np.full(T, 1.2, dtype=np.float32)
    for t in range(T):
        ys = []
        if np.any(np.abs(sub[t, 33:54]) > 1e-6):
            ys.append(float(sub[t, 33, 1]))
        if np.any(np.abs(sub[t, 54:75]) > 1e-6):
            ys.append(float(sub[t, 54, 1]))
        if ys:
            wy[t] = min(ys)

    y_top = float(np.percentile(wy, 25))
    y_span = max(0.06, float(np.percentile(wy, 90)) - y_top)
    thresh_y = min(0.92, max(y_top + 0.55 * y_span, y_top + 0.08))
    a_idx = np.where(wy <= thresh_y)[0]
    core = sub
    if len(a_idx) >= 8:
        s = max(0, int(a_idx[0]) - 1)
        e = min(T, int(a_idx[-1]) + 2)
        if e - s >= 8:
            core = sub[s:e]

    # If a clip was accidentally recorded from a long sentence buffer (>48 frames), isolate the 36-frame peak motion sign
    if len(core) > 48:
        vel = np.linalg.norm(np.diff(core[:, 33:75, :2], axis=0), axis=(1, 2))
        w_conv = min(30, len(vel))
        best_s = int(np.argmax(np.convolve(vel, np.ones(w_conv), mode="valid")))
        core = core[best_s : min(len(core), best_s + 36)]

    wins: List[np.ndarray] = [core]
    if len(sub) <= 65 and len(sub) != len(core):
        wins.append(sub)
    C = len(core)
    if C >= 16:
        wins.append(core[: max(10, int(C * 0.78))])
        wins.append(core[min(C - 10, int(C * 0.22)) :])
    if fast_mode:
        if C >= 22:
            wins.append(core[(C - 18) // 2 : (C - 18) // 2 + 18])
        return wins
    if C >= 22:
        wins.append(core[int(C * 0.12) : int(C * 0.88)])
    for wlen in (14, 22, 30):
        if C >= wlen + 4:
            wins.append(core[:wlen])
            wins.append(core[(C - wlen) // 2 : (C - wlen) // 2 + wlen])
            wins.append(core[-wlen:])
    return wins


def seq138_to_descriptor(seq_30x138: Any) -> np.ndarray:
    """
    Build a speed-tolerant, L2-normalized descriptor from a `(30, 138)` sign feature matrix
    for few-shot webcam prototype matching. Emphasizes handshape (`0:102`) and upper-body XY
    location (`106:122, 128:138`) while down-weighting noisy Z-depth and raw velocity spikes.
    """
    arr = np.asarray(seq_30x138, dtype=np.float32)
    w = np.ones(138, dtype=np.float32)
    w[102:106] = 0.35
    w[111] = 0.15
    w[117] = 0.15
    w[122:128] = 0.35
    aw = arr * w[None, :]
    p1 = aw[:10].mean(axis=0)
    p2 = aw[10:20].mean(axis=0)
    p3 = aw[20:].mean(axis=0)
    gm = aw.mean(axis=0)
    gs = aw.std(axis=0)
    traj = aw.reshape(10, 3, 138).mean(axis=1).reshape(-1) * 0.35
    desc = np.concatenate([p1, p2, p3, gm, gs, traj], axis=0)
    norm = float(np.linalg.norm(desc))
    return desc / max(norm, 1e-6)


class DualStudioState:
    def __init__(self) -> None:
        os.makedirs(DEFAULT_CKPT_DIR, exist_ok=True)
        os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
        os.makedirs(WEBCAM_NPY_DIR, exist_ok=True)
        os.makedirs(WEBCAM_SENTENCE_NPY_DIR, exist_ok=True)
        self.vocab = Vocabulary()
        self.default_npy_root = (
            find_default_npy_root()
            or r"C:\VSL400\vsl400_video_extracted\keypoints_spliteds"
        )

        # Part 1: Isolated Word Recognition (BiLSTM, 30x138) — STRICTLY SEPARATED FROM PART 2
        self.word_recognizer: Optional[VSLSingleSignRecognizer] = None
        self.word_is_demo: bool = False
        self.word_live_buffer_76x3: List[np.ndarray] = []
        self.word_missing_hand_streak: int = 0
        self.word_step_counter: int = 0
        self.word_ema_probs: Optional[np.ndarray] = None
        self.word_peak_prediction: Optional[Dict[str, Any]] = None
        self.last_word_prediction: Optional[Dict[str, Any]] = None
        self.last_captured_clip_76x3: Optional[np.ndarray] = None
        self.word_replay_cache: Optional[List[Any]] = None
        self.webcam_prototypes: Dict[str, List[np.ndarray]] = {}
        self.webcam_file_cache: Dict[str, Dict[str, Any]] = {}

        # Part 2: Continuous Sentence Recognition — STRICTLY SEPARATED FROM PART 1
        # Uses its own dedicated `sentence_segment_recognizer` (vsl_sentence_segment.pth)
        # and `sentence_recognizer` (vsl_sentence_ctc.pth) trained ONLY on Part 2's vocabulary.
        self.sentence_segment_recognizer: Optional[VSLSingleSignRecognizer] = None
        self.sentence_recognizer: Optional[VSLContinuousRecognizer] = None
        self.sentence_stream_engine: Optional[CTCStreamInferenceEngine] = None
        self.word_spotting_decoder: Optional[WordSpottingSentenceDecoder] = None
        self.sentence_abrupt_tracker: Optional[ContinuousAbruptStreamTracker] = None
        self.word_grammar: Optional[SentenceGrammar] = None
        self.sentence_replay_cache: Optional[List[Any]] = None
        self.sentence_is_demo: bool = False
        self.sentence_live_buffer_76x3: List[np.ndarray] = []
        self.sentence_missing_hand_streak: int = 0
        self.sentence_step_counter: int = 0
        self.sentence_ctc_new_frames: int = 0
        self.sentence_ctc_done: bool = False
        self.last_sentence_prediction: Optional[Dict[str, Any]] = None

        self._load_existing_checkpoints()
        self._sync_webcam_prototypes_from_disk()

    def get_webcam_sample_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if not os.path.isdir(WEBCAM_NPY_DIR):
            return counts
        for cls in os.listdir(WEBCAM_NPY_DIR):
            p = os.path.join(WEBCAM_NPY_DIR, cls)
            if os.path.isdir(p):
                n = sum(1 for fn in os.listdir(p) if fn.endswith(".npy"))
                if n > 0:
                    counts[normalize_single_gloss(cls)] = n
        return counts

    def _sync_webcam_prototypes_from_disk(self) -> None:
        """Load descriptors of saved `data/webcam_npy/<cls>/*.npy` clips into `self.webcam_prototypes`."""
        self.webcam_prototypes = {}
        if not os.path.isdir(WEBCAM_NPY_DIR):
            return
        for cls in os.listdir(WEBCAM_NPY_DIR):
            cls_dir = os.path.join(WEBCAM_NPY_DIR, cls)
            if not os.path.isdir(cls_dir):
                continue
            norm_cls = normalize_single_gloss(cls)
            files = sorted(
                os.path.join(cls_dir, fn) for fn in os.listdir(cls_dir) if fn.endswith(".npy")
            )[-8:]
            if not files:
                continue
            descs: List[np.ndarray] = []
            for fpath in files:
                cached = self.webcam_file_cache.get(fpath)
                if cached is not None:
                    descs.extend(cached.get("descs", []))
                    continue
                try:
                    w_arr = np.load(fpath, allow_pickle=False)
                except Exception:
                    continue
                if len(w_arr) > 85 and len(files) > 1:
                    continue
                wins = extract_active_sign_windows_76x3(w_arr, fast_mode=True)
                f_seqs = []
                f_descs = []
                for w in wins[:4]:
                    s138 = npy_sequence_to_word_138_features(w, target_len=SINGLE_SIGN_SEQUENCE_LENGTH)
                    f_seqs.append(s138)
                    f_descs.append(seq138_to_descriptor(s138))
                self.webcam_file_cache[fpath] = {
                    "label": norm_cls,
                    "seqs": f_seqs,
                    "aug_seqs": [],
                    "descs": f_descs,
                }
                descs.extend(f_descs)
            if descs:
                self.webcam_prototypes[norm_cls] = descs

    def get_word_classes(self) -> List[str]:
        return list(self.word_recognizer.class_names) if self.word_recognizer is not None else []

    def get_sentence_classes(self) -> List[str]:
        if self.sentence_segment_recognizer is not None:
            return list(self.sentence_segment_recognizer.class_names)
        if self.sentence_recognizer is not None:
            return [w for k, w in sorted(self.sentence_recognizer.vocab.id_to_word.items()) if k != 0]
        return []

    def _rebuild_sentence_spotting(self) -> None:
        if self.sentence_segment_recognizer is not None:
            lm, grammar = build_language_model_for_vocabulary(self.sentence_segment_recognizer.class_names)
            self.word_grammar = grammar
            self.word_spotting_decoder = WordSpottingSentenceDecoder(
                self.sentence_segment_recognizer,
                lm=lm,
                pos_step=3,
                min_sign_frames=6,
                skip_penalty_per_frame=-0.22,
                ctc_recognizer=self.sentence_recognizer,
            )
            self.sentence_abrupt_tracker = ContinuousAbruptStreamTracker(
                self.sentence_segment_recognizer,
                lm=lm,
                min_new_frames_for_preview=3,
                ctc_recognizer=self.sentence_recognizer,
            )

    def _load_existing_checkpoints(self) -> None:
        if os.path.exists(DEFAULT_WORD_CKPT):
            try:
                self.word_recognizer = VSLSingleSignRecognizer(DEFAULT_WORD_CKPT)
            except Exception as e:
                print(f"[SERVER] Could not load word checkpoint: {e}")

        if os.path.exists(DEFAULT_SENTENCE_CKPT):
            try:
                self.sentence_recognizer = VSLContinuousRecognizer(DEFAULT_SENTENCE_CKPT)
                self.sentence_stream_engine = CTCStreamInferenceEngine(
                    self.sentence_recognizer,
                    window_size_frames=75,
                    stride_frames=6,
                    min_frames_to_infer=12,
                )
                self.vocab = self.sentence_recognizer.vocab
                if self.word_grammar is None:
                    self.word_grammar = self.sentence_recognizer.grammar
            except Exception as e:
                print(f"[SERVER] Could not load sentence checkpoint: {e}")

        if os.path.exists(DEFAULT_SENTENCE_SEGMENT_CKPT):
            try:
                self.sentence_segment_recognizer = VSLSingleSignRecognizer(DEFAULT_SENTENCE_SEGMENT_CKPT)
                self._rebuild_sentence_spotting()
            except Exception as e:
                print(f"[SERVER] Could not load sentence segment checkpoint: {e}")

    def reload_word_checkpoint(self, path: str, is_demo: bool = False) -> None:
        self.word_recognizer = VSLSingleSignRecognizer(path)
        self.word_is_demo = is_demo
        self.word_ema_probs = None
        self.word_replay_cache = None

    def reload_sentence_checkpoint(
        self,
        ctc_path: str,
        segment_path: Optional[str] = None,
        is_demo: bool = False,
    ) -> None:
        self.sentence_recognizer = VSLContinuousRecognizer(ctc_path)
        self.vocab = self.sentence_recognizer.vocab
        self.sentence_stream_engine = CTCStreamInferenceEngine(
            self.sentence_recognizer,
            window_size_frames=75,
            stride_frames=6,
            min_frames_to_infer=12,
        )
        if segment_path and os.path.exists(segment_path):
            self.sentence_segment_recognizer = VSLSingleSignRecognizer(segment_path)
            self.sentence_replay_cache = None
        self._rebuild_sentence_spotting()
        self.sentence_is_demo = is_demo

    def ensure_word_model_loaded(self) -> None:
        if self.word_recognizer is None:
            if os.path.isdir(self.default_npy_root):
                word_samples = load_word_samples_from_npy_root(
                    root_dir=self.default_npy_root,
                    max_classes=8,
                    max_files_per_class_split=35,
                )
                train_single_sign_pipeline(
                    rows=word_samples,
                    output_model_path=DEFAULT_WORD_CKPT,
                    epochs=35,
                    batch_size=8,
                    lr=1.5e-3,
                    seed=42,
                )
                self.reload_word_checkpoint(DEFAULT_WORD_CKPT, is_demo=False)
            else:
                _, single_samples = build_pipeline_verification_dataset()
                train_single_sign_pipeline(
                    rows=single_samples,
                    output_model_path=DEFAULT_WORD_CKPT,
                    epochs=4,
                    batch_size=4,
                    seed=42,
                )
                self.reload_word_checkpoint(DEFAULT_WORD_CKPT, is_demo=True)

    def ensure_sentence_model_loaded(self) -> None:
        if self.sentence_recognizer is None or self.sentence_segment_recognizer is None:
            if os.path.isdir(self.default_npy_root):
                target_folders = self.get_word_classes() or None
                n_cls = len(target_folders) if target_folders else 12
                seg_samples = load_word_samples_from_npy_root(
                    root_dir=self.default_npy_root,
                    selected_folders=target_folders,
                    max_classes=n_cls,
                    max_files_per_class_split=90,
                )
                train_single_sign_pipeline(
                    rows=seg_samples,
                    output_model_path=DEFAULT_SENTENCE_SEGMENT_CKPT,
                    epochs=45,
                    batch_size=8,
                    lr=1.5e-3,
                    seed=42,
                )
                samples, custom_vocab = build_continuous_sentence_samples_from_npy_root(
                    root_dir=self.default_npy_root,
                    selected_folders=target_folders,
                    max_classes=n_cls,
                    sentences_per_split=100,
                    use_grammar=True,
                )
                train_ctc_pipeline(
                    raw_samples=samples,
                    output_model_path=DEFAULT_SENTENCE_CKPT,
                    vocab=custom_vocab,
                    epochs=25,
                    batch_size=8,
                    lr=1.5e-3,
                    seed=42,
                )
                self.reload_sentence_checkpoint(
                    DEFAULT_SENTENCE_CKPT, DEFAULT_SENTENCE_SEGMENT_CKPT, is_demo=False
                )
            else:
                ctc_samples, single_samples = build_pipeline_verification_dataset()
                train_single_sign_pipeline(
                    rows=single_samples,
                    output_model_path=DEFAULT_SENTENCE_SEGMENT_CKPT,
                    epochs=4,
                    batch_size=4,
                    seed=42,
                )
                train_ctc_pipeline(
                    raw_samples=ctc_samples,
                    output_model_path=DEFAULT_SENTENCE_CKPT,
                    vocab=self.vocab,
                    epochs=3,
                    batch_size=4,
                    seed=42,
                )
                self.reload_sentence_checkpoint(
                    DEFAULT_SENTENCE_CKPT, DEFAULT_SENTENCE_SEGMENT_CKPT, is_demo=True
                )

    def reset_word_stream(self) -> None:
        self.word_live_buffer_76x3 = []
        self.word_missing_hand_streak = 0
        self.word_step_counter = 0
        self.word_ema_probs = None
        self.word_peak_prediction = None
        self.last_word_prediction = None

    def reset_sentence_stream(self) -> None:
        if self.sentence_stream_engine is not None:
            self.sentence_stream_engine.reset()
        if self.sentence_abrupt_tracker is not None:
            self.sentence_abrupt_tracker.reset()
        self.sentence_live_buffer_76x3 = []
        self.sentence_missing_hand_streak = 0
        self.sentence_step_counter = 0
        self.sentence_ctc_new_frames = 0
        self.sentence_ctc_done = False
        self.last_sentence_prediction = None


STATE = DualStudioState()


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VSL Recognition Studio — Hệ thống Nhận diện Ngôn ngữ Ký hiệu Việt Nam</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js" crossorigin="anonymous"></script>
  <style>
    :root {
      --bg: #090d16;
      --bg-grad: radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.08), transparent 40%),
                 radial-gradient(circle at 85% 15%, rgba(168, 85, 247, 0.08), transparent 40%);
      --panel: rgba(19, 28, 46, 0.85);
      --panel-solid: #131c2e;
      --panel-alt: #0d1422;
      --border: rgba(148, 163, 184, 0.16);
      --border-strong: rgba(148, 163, 184, 0.28);
      --text: #f8fafc;
      --text-soft: #cbd5e1;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-dark: #0284c7;
      --purple: #a855f7;
      --purple-light: #c084fc;
      --success: #10b981;
      --warn: #f59e0b;
      --danger: #ef4444;
      --radius-lg: 16px;
      --radius-md: 12px;
      --radius-sm: 8px;
      --shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.55);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background-color: var(--bg);
      background-image: var(--bg-grad);
      background-attachment: fixed;
      color: var(--text);
      line-height: 1.55;
      min-height: 100vh;
    }
    header {
      padding: 16px 28px;
      background: rgba(9, 13, 22, 0.88);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 40;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .brand-wrap {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-icon {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.35rem;
      box-shadow: 0 4px 15px rgba(56, 189, 248, 0.25);
      flex-shrink: 0;
    }
    header h1 {
      margin: 0;
      font-size: 1.18rem;
      font-weight: 800;
      letter-spacing: -0.01em;
      background: linear-gradient(90deg, #f8fafc, #7dd3fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .header-sub {
      font-size: 0.8rem;
      color: var(--muted);
      margin-top: 2px;
    }
    .status-pills {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      font-size: 0.82rem;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 6px 13px;
      border-radius: 999px;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      color: var(--text-soft);
      font-weight: 500;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #64748b;
    }
    .status-dot.online {
      background: var(--success);
      box-shadow: 0 0 8px var(--success);
    }

    /* TOP DATASET BAR */
    .top-bar-wrap {
      max-width: 1380px;
      margin: 18px auto 0;
      padding: 0 22px;
    }
    .dataset-bar {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 12px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      box-shadow: var(--shadow);
    }
    .dataset-bar-left {
      display: flex;
      align-items: center;
      gap: 10px;
      flex: 1;
      min-width: 280px;
    }
    .dataset-badge {
      padding: 5px 10px;
      border-radius: 7px;
      background: rgba(56, 189, 248, 0.12);
      color: var(--accent);
      font-weight: 700;
      font-size: 0.78rem;
      white-space: nowrap;
    }

    /* MODE SWITCHER TABS */
    .tabs {
      max-width: 1380px;
      margin: 16px auto 0;
      padding: 0 22px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    @media (max-width: 820px) {
      .tabs { grid-template-columns: 1fr; }
    }
    .tab-btn {
      padding: 16px 20px;
      border-radius: var(--radius-lg);
      border: 1.5px solid var(--border);
      background: var(--panel);
      color: var(--text-soft);
      cursor: pointer;
      text-align: left;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      align-items: flex-start;
      gap: 14px;
      box-shadow: var(--shadow);
    }
    .tab-btn:hover {
      border-color: var(--border-strong);
      transform: translateY(-1px);
    }
    .tab-icon {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
      background: rgba(148, 163, 184, 0.1);
      flex-shrink: 0;
    }
    .tab-title {
      font-size: 0.98rem;
      font-weight: 700;
      color: var(--text);
      display: block;
    }
    .tab-sub {
      display: block;
      font-size: 0.81rem;
      font-weight: 400;
      color: var(--muted);
      margin-top: 3px;
    }
    .tab-btn.active-word {
      border-color: var(--accent);
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(19, 28, 46, 0.92));
      box-shadow: 0 8px 25px -6px rgba(56, 189, 248, 0.25);
    }
    .tab-btn.active-word .tab-icon {
      background: rgba(56, 189, 248, 0.2);
      color: var(--accent);
    }
    .tab-btn.active-sentence {
      border-color: var(--purple);
      background: linear-gradient(135deg, rgba(168, 85, 247, 0.16), rgba(19, 28, 46, 0.92));
      box-shadow: 0 8px 25px -6px rgba(168, 85, 247, 0.25);
    }
    .tab-btn.active-sentence .tab-icon {
      background: rgba(168, 85, 247, 0.22);
      color: var(--purple-light);
    }

    /* MAIN GRID */
    .container {
      max-width: 1380px;
      margin: 0 auto;
      padding: 18px 22px 24px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    @media (max-width: 1020px) {
      .container { grid-template-columns: 1fr; }
    }
    .hidden { display: none !important; }

    /* CARDS */
    .card {
      background: var(--panel);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 20px 22px;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
    }
    .card h2 {
      margin: 0;
      font-size: 1.04rem;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 9px;
    }
    .step-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .sentence-card .step-badge {
      background: rgba(168, 85, 247, 0.16);
      color: var(--purple-light);
      border-color: rgba(168, 85, 247, 0.32);
    }

    /* FORM CONTROLS */
    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
      gap: 10px;
    }
    .field {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .field label {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--muted);
    }
    input[type="number"], input[type="text"], select, input[type="file"] {
      background: var(--panel-alt);
      color: var(--text);
      border: 1px solid var(--border-strong);
      padding: 8px 11px;
      border-radius: var(--radius-sm);
      font-family: inherit;
      font-size: 0.86rem;
      transition: border-color 0.15s, box-shadow 0.15s;
      width: 100%;
    }
    input[type="number"]:focus, input[type="text"]:focus, select:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
    }
    .row {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }

    /* BUTTONS */
    button, .btn-link {
      background: linear-gradient(135deg, #38bdf8, #0284c7);
      color: #ffffff;
      border: none;
      padding: 9px 16px;
      border-radius: var(--radius-sm);
      font-family: inherit;
      font-size: 0.86rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
      transition: all 0.15s ease;
      text-decoration: none;
    }
    button:hover:not(:disabled), .btn-link:hover {
      filter: brightness(1.08);
      transform: translateY(-1px);
    }
    button.purple {
      background: linear-gradient(135deg, #a855f7, #7e22ce);
      color: #ffffff;
    }
    button.green {
      background: linear-gradient(135deg, #10b981, #059669);
      color: #ffffff;
    }
    button.secondary, .btn-link.secondary {
      background: rgba(51, 65, 85, 0.65);
      color: var(--text-soft);
      border: 1px solid var(--border-strong);
    }
    button.secondary:hover:not(:disabled), .btn-link.secondary:hover {
      background: rgba(71, 85, 105, 0.8);
      color: #ffffff;
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      transform: none;
    }

    /* INFO & RESULT BOXES */
    .hint-box {
      background: rgba(13, 20, 34, 0.7);
      border-left: 3px solid var(--accent);
      border-radius: 6px;
      padding: 10px 13px;
      font-size: 0.82rem;
      color: var(--text-soft);
    }
    .sentence-card .hint-box {
      border-left-color: var(--purple);
    }
    .result-box {
      padding: 14px 16px;
      background: var(--panel-alt);
      border-radius: var(--radius-md);
      border: 1px solid var(--border);
    }
    .result-label {
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin-bottom: 8px;
    }

    /* PREDICTION BADGES & BARS */
    .pred-hero {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.02);
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .word-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 16px;
      border-radius: 9px;
      background: var(--accent-dark);
      color: white;
      font-weight: 800;
      font-size: 1.15rem;
      letter-spacing: 0.01em;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .top3-list {
      margin-top: 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .top3-item {
      display: grid;
      grid-template-columns: 110px 1fr 52px;
      align-items: center;
      gap: 8px;
      font-size: 0.8rem;
    }
    .top3-bar-bg {
      height: 7px;
      background: rgba(148, 163, 184, 0.14);
      border-radius: 999px;
      overflow: hidden;
    }
    .top3-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #38bdf8, #10b981);
      border-radius: 999px;
      transition: width 0.25s ease;
    }
    .gloss-chip {
      display: inline-flex;
      align-items: center;
      padding: 6px 13px;
      margin: 3px;
      background: linear-gradient(135deg, #9333ea, #6b21a8);
      color: white;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.94rem;
      box-shadow: 0 3px 10px rgba(147, 51, 234, 0.3);
    }

    /* COLLAPSIBLE TECHNICAL DETAILS */
    details.tech-details {
      background: rgba(9, 13, 22, 0.55);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      overflow: hidden;
    }
    details.tech-details summary {
      padding: 8px 12px;
      font-size: 0.79rem;
      font-weight: 600;
      color: var(--muted);
      cursor: pointer;
      user-select: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    details.tech-details summary:hover {
      color: var(--text-soft);
      background: rgba(255, 255, 255, 0.02);
    }
    pre {
      margin: 0;
      background: #060911;
      padding: 12px;
      border-top: 1px solid var(--border);
      overflow-x: auto;
      font-size: 0.78rem;
      color: #94a3b8;
      max-height: 240px;
      overflow-y: auto;
    }

    /* WEBCAM STUDIO */
    .webcam-grid {
      display: grid;
      grid-template-columns: 460px 1fr;
      gap: 22px;
      align-items: start;
    }
    @media (max-width: 980px) {
      .webcam-grid { grid-template-columns: 1fr; }
    }
    .video-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 4/3;
      background: #050811;
      border-radius: var(--radius-md);
      overflow: hidden;
      border: 1.5px solid var(--border-strong);
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.6);
    }
    video { display: none; }
    canvas { width: 100%; height: 100%; object-fit: cover; display: block; }
    .cam-overlay-bar {
      position: absolute;
      bottom: 10px;
      left: 10px;
      right: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      pointer-events: none;
    }
    .hand-pill {
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 700;
      background: rgba(9, 13, 22, 0.78);
      backdrop-filter: blur(6px);
      border: 1px solid var(--border);
      color: var(--muted);
      transition: all 0.15s;
    }
    .hand-pill.active-left {
      border-color: #22c55e;
      color: #4ade80;
      background: rgba(22, 163, 74, 0.22);
    }
    .hand-pill.active-right {
      border-color: #38bdf8;
      color: #7dd3fc;
      background: rgba(2, 132, 199, 0.22);
    }

    /* CONFUSION MATRIX TABLES */
    .cm-layout {
      display: grid;
      grid-template-columns: 1fr 420px;
      gap: 20px;
      align-items: start;
    }
    @media (max-width: 1080px) {
      .cm-layout { grid-template-columns: 1fr; }
    }
    .cm-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
      background: var(--panel-alt);
      text-align: center;
    }
    .cm-table th, .cm-table td {
      border: 1px solid rgba(148, 163, 184, 0.14);
      padding: 7px 9px;
      white-space: nowrap;
    }
    .cm-table th {
      background: #0f172a;
      color: #94a3b8;
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    .cm-table th.cm-row-header {
      position: sticky;
      left: 0;
      z-index: 3;
      text-align: left;
      background: #0f172a;
      color: #f8fafc;
      font-weight: 700;
    }
    .cm-cell {
      transition: all 0.1s ease-in-out;
      cursor: pointer;
    }
    .cm-cell:hover {
      outline: 2px solid var(--accent);
      transform: scale(1.04);
      z-index: 4;
      position: relative;
    }
    .cm-metrics-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .cm-metrics-table th, .cm-metrics-table td {
      border: 1px solid rgba(148, 163, 184, 0.12);
      padding: 7px 11px;
      text-align: left;
    }
    .cm-metrics-table th {
      background: #0f172a;
      color: #94a3b8;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand-wrap">
      <div class="brand-icon">🤟</div>
      <div>
        <h1>VSL Recognition Studio</h1>
        <div class="header-sub">
          Hệ thống Nhận diện Ngôn ngữ Ký hiệu Việt Nam — Từ đơn (BiLSTM) &amp; Câu liên tục (CTC)
        </div>
      </div>
    </div>
    <div id="headerStats" class="status-pills">
      <span class="status-pill"><span class="status-dot"></span> Đang kiểm tra trạng thái mô hình...</span>
    </div>
  </header>

  <!-- CONFIGURATION BAR FOR KEYPOINTS_SPLITED ROOT -->
  <div class="top-bar-wrap">
    <div class="dataset-bar">
      <div class="dataset-bar-left">
        <span class="dataset-badge">📁 Nguồn dữ liệu (.npy)</span>
        <input type="text" id="npyRootInput" style="flex:1; min-width:260px;" value="C:\VSL400\SignCTCModel\archive\processed\processed\keypoints_splited" placeholder="Đường dẫn thư mục keypoints_splited..." />
        <button onclick="scanNpyFolder()" class="secondary">🔍 Quét dữ liệu</button>
      </div>
      <span id="npyScanSummary" style="font-size:0.83rem; color:var(--accent); font-weight:500;"></span>
    </div>
  </div>

  <!-- TWO MAIN TABS -->
  <div class="tabs">
    <button id="tabBtnWord" class="tab-btn active-word" onclick="switchMode('word')">
      <div class="tab-icon">🔤</div>
      <div>
        <span class="tab-title">Chế độ 1: Nhận diện Từ đơn (Isolated Word — BiLSTM)</span>
        <span class="tab-sub">Phân loại từng ký hiệu từ vựng độc lập từ file <code>.npy</code> hoặc Webcam trực tiếp</span>
      </div>
    </button>
    <button id="tabBtnSentence" class="tab-btn" onclick="switchMode('sentence')">
      <div class="tab-icon">💬</div>
      <div>
        <span class="tab-title">Chế độ 2: Nhận diện Câu liên tục (Continuous Sentence — BiLSTM + CTC)</span>
        <span class="tab-sub">Giải mã chuỗi nhiều từ ký hiệu nối tiếp nhau không cần cắt đoạn thủ công</span>
      </div>
    </button>
  </div>

  <!-- ===================================================================== -->
  <!-- PART 1: ISOLATED WORD RECOGNITION (BiLSTM)                            -->
  <!-- ===================================================================== -->
  <div id="panelWord" class="container">
    <!-- 1A. TRAIN WORD -->
    <div class="card">
      <div class="card-header">
        <h2><span class="step-badge">Bước 1</span> Huấn luyện Mô hình Từ đơn</h2>
        <span style="font-size:0.78rem; color:var(--muted);">Kiến trúc: BiLSTM (30 &times; 138)</span>
      </div>

      <div class="hint-box">
        Chọn danh sách các từ ký hiệu bạn muốn dạy cho máy (hoặc để trống để hệ thống tự chọn các từ có nhiều mẫu nhất trong thư mục <code>keypoints_splited</code>).
      </div>

      <div class="field">
        <label>Danh sách từ muốn huấn luyện (cách nhau bởi dấu phẩy, để trống = tự động):</label>
        <input type="text" id="wordSelectedFolders" placeholder="Ví dụ: An ủi, Anh, Ba lô, Ban ngày, Ban đêm (Để trống sẽ tự chọn N từ đầu tiên)" />
      </div>

      <div class="form-grid">
        <div class="field">
          <label>Số lượng từ (Classes)</label>
          <input type="number" id="wordMaxClasses" value="8" min="2" max="472" />
        </div>
        <div class="field">
          <label>Số mẫu tối đa / từ</label>
          <input type="number" id="wordMaxFiles" value="35" min="5" max="200" />
        </div>
        <div class="field">
          <label>Số vòng lặp (Epochs)</label>
          <input type="number" id="wordEpochs" value="35" min="1" max="300" />
        </div>
        <div class="field">
          <label>Batch Size</label>
          <input type="number" id="wordBatch" value="8" min="1" max="128" />
        </div>
        <div class="field">
          <label>Tốc độ học (LR)</label>
          <input type="number" id="wordLr" value="0.0015" step="0.0005" />
        </div>
      </div>

      <div class="row">
        <button id="btnTrainWordNpy" class="green" onclick="trainWordFromNpy()" style="width:100%; padding:11px;">
          🚀 Bắt đầu Huấn luyện Mô hình Từ đơn
        </button>
      </div>

      <div id="wordTrainMsg" style="font-size:0.86rem; color:var(--accent); font-weight:500;"></div>

      <details class="tech-details">
        <summary><span>⚙️ Xem chi tiết kỹ thuật &amp; Nhật ký huấn luyện (JSON)</span><span>▾</span></summary>
        <pre id="wordTrainOutput">Chưa có nhật ký huấn luyện. Hãy bấm "Bắt đầu Huấn luyện Mô hình Từ đơn".</pre>
      </details>
    </div>

    <!-- 1B. TEST / PREDICT WORD -->
    <div class="card">
      <div class="card-header">
        <h2><span class="step-badge">Bước 2</span> Kiểm thử Nhận diện Từ đơn từ File</h2>
        <span style="font-size:0.78rem; color:var(--muted);">Đầu vào: File <code>.npy</code></span>
      </div>

      <div class="hint-box">
        Tải lên 1 file <code>.npy</code> từ máy tính hoặc lấy ngẫu nhiên 1 mẫu trong tập <code>test/</code> để kiểm tra độ chính xác của mô hình.
      </div>

      <div class="field">
        <label>Tải lên 1 file <code>.npy</code> từ máy tính:</label>
        <div class="row">
          <input type="file" id="wordNpyFileInput" accept=".npy" style="flex:1;" />
          <button onclick="predictWordFromUploadedNpy()">🔍 Nhận diện File</button>
        </div>
      </div>

      <div class="row">
        <button class="secondary" onclick="predictRandomTestNpyWord()" style="width:100%;">
          🎲 Lấy ngẫu nhiên 1 mẫu trong Tập Test để thử nghiệm
        </button>
      </div>

      <div class="result-box">
        <div class="result-label">Kết quả Dự đoán (BiLSTM)</div>
        <div id="wordSampleBadge" style="color:var(--muted); font-size:0.9rem;">
          Hãy chọn 1 file <code>.npy</code> hoặc bấm nút lấy mẫu ngẫu nhiên ở trên.
        </div>
      </div>

      <details class="tech-details">
        <summary><span>📊 Xem chi tiết vector xác suất (JSON)</span><span>▾</span></summary>
        <pre id="wordPredictOutput">Chưa có dữ liệu dự đoán.</pre>
      </details>
    </div>

    <!-- 1C. CONFUSION MATRIX VISUALIZATION FOR WORD BILSTM -->
    <div class="card" id="wordCmCard" style="grid-column: 1 / -1; display:none;">
      <div class="card-header" style="flex-wrap:wrap;">
        <h2><span class="step-badge">Đánh giá</span> Ma trận nhầm lẫn (Confusion Matrix) — Mô hình Từ đơn</h2>
        <div class="row">
          <button class="secondary" onclick="refreshWordConfusionMatrix(true)">🔄 Tính toán lại trên Tập Test</button>
          <button id="btnToggleWordNorm" class="secondary" onclick="toggleWordCmNormalize()">Chế độ: Tỉ lệ chuẩn hoá %</button>
          <a id="btnDownloadWordCm" class="btn-link secondary" download="vsl_word_confusion_matrix.png">⬇️ Tải ảnh PNG</a>
        </div>
      </div>
      <div id="wordCmSummaryText" style="font-size:0.92rem; color:var(--text-soft);">Đang tải ma trận nhầm lẫn...</div>
      <div class="cm-layout">
        <div style="min-width:0;">
          <div style="font-size:0.82rem; color:var(--muted); margin-bottom:6px;">
            Bảng nhiệt tương tác — <em>Hàng ngang: Nhãn thực tế (True) &times; Cột dọc: Nhãn dự đoán (Pred)</em>
          </div>
          <div id="wordCmTableContainer" style="overflow-x:auto; max-height:440px; overflow-y:auto; border:1px solid var(--border); border-radius:10px; margin-bottom:14px;"></div>
          <div style="font-size:0.82rem; color:var(--muted); margin-bottom:6px;">Chỉ số đánh giá chi tiết từng từ (Precision, Recall, F1-Score):</div>
          <div id="wordCmMetricsContainer" style="overflow-x:auto;"></div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:0.82rem; color:var(--muted); margin-bottom:6px;">Biểu đồ trực quan (Matplotlib Heatmap):</div>
          <img id="wordCmImg" style="width:100%; max-width:420px; border-radius:10px; border:1px solid var(--border); background:#060911;" alt="Word Confusion Matrix" />
        </div>
      </div>
    </div>
  </div>

  <!-- ===================================================================== -->
  <!-- PART 2: CONTINUOUS SENTENCE RECOGNITION (BiLSTM + CTC & WORD-SPOTTING)-->
  <!-- ===================================================================== -->
  <div id="panelSentence" class="container hidden">
    <!-- 2A. TRAIN SENTENCE -->
    <div class="card sentence-card">
      <div class="card-header">
        <h2><span class="step-badge">Bước 1</span> Biến tấu &amp; Huấn luyện Mô hình Câu (CTC + Ngữ pháp)</h2>
        <span style="font-size:0.78rem; color:var(--muted);">2-Layer BiLSTM + CTCLoss + Bigram LM</span>
      </div>

      <div class="hint-box">
        <strong>💡 Hệ thống Nhận diện Câu Liên tục Độc lập (Chuyển ký hiệu đột ngột không cần hạ tay):</strong><br/>
        • <strong>🧩 2 Hệ thống Mô hình Độc lập:</strong> Phần 2 tự huấn luyện và lưu bộ trọng số riêng (<code>vsl_sentence_segment.pth</code> + <code>vsl_sentence_ctc.pth</code>), hoàn toàn độc lập với mô hình Từ đơn (Phần 1). Bên Câu (CTC) có thể tự train lại trên cùng bộ dữ liệu <code>.npy</code> đó hoặc bất kỳ danh sách từ nào bạn chọn.<br/>
        • <strong>⚡ Bắt chuyển động đột ngột:</strong> Kết hợp Bộ phân đoạn Chuyển tiếp Đột ngột + Mạng 2-Layer BiLSTM CTC + Ngữ pháp Tiếng Việt Bigram LM, nhận diện mượt khi người ký hiệu chuyển từ liên tiếp mà không cần hạ tay reset.
      </div>

      <div class="field">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;">
          <label>Danh sách các từ muốn huấn luyện bên Câu / CTC (để trống = tự lấy theo danh sách từ hiện có):</label>
          <button type="button" class="secondary" onclick="syncFoldersFromWordModel()" style="padding:4px 10px; font-size:0.76rem;">
            🔗 Điền danh sách từ giống bên Từ đơn để CTC tự train lại
          </button>
        </div>
        <input type="text" id="sentSelectedFolders" placeholder="Ví dụ: TÔI, BÁC_SĨ, BỆNH_VIỆN, ĐAU, ĐẦU, SỐT, HO, MỆT, LẠNH, NÔN_ÓI, BIẾT, BÂY_GIỜ" />
      </div>

      <div class="form-grid">
        <div class="field">
          <label>Số lượng từ vựng</label>
          <input type="number" id="sentMaxClasses" value="12" min="2" max="100" />
        </div>
        <div class="field">
          <label>Số câu sinh ra</label>
          <input type="number" id="sentCount" value="140" min="15" max="600" />
        </div>
        <div class="field">
          <label>Số vòng lặp (Epochs)</label>
          <input type="number" id="sentenceEpochs" value="30" min="1" max="300" />
        </div>
        <div class="field">
          <label>Batch Size</label>
          <input type="number" id="sentenceBatch" value="8" min="1" max="128" />
        </div>
      </div>

      <label style="display:flex; align-items:center; gap:8px; font-size:0.83rem; color:var(--text-soft); cursor:pointer; background:var(--panel-alt); padding:8px 12px; border-radius:8px; border:1px solid var(--border);">
        <input type="checkbox" id="chkUseSentenceGrammar" checked />
        <span><strong>Sinh câu theo Ngữ pháp Tiếng Việt</strong> (kèm huấn luyện Mô hình Ngôn ngữ Bigram LM tự động)</span>
      </label>

      <div id="grammarPreviewBox" class="result-box" style="padding:10px 13px; font-size:0.8rem; color:var(--text-soft);">
        Đang tải mẫu câu ngữ pháp...
      </div>

      <div class="row">
        <button id="btnTrainSentNpy" class="purple" onclick="trainSentenceFromNpy()" style="width:100%; padding:11px;">
          🚀 Bắt đầu Sinh câu Ngữ pháp &amp; Huấn luyện Mô hình Câu (CTC)
        </button>
      </div>

      <div id="sentenceTrainMsg" style="font-size:0.86rem; color:var(--purple-light); font-weight:500;"></div>

      <details class="tech-details">
        <summary><span>⚙️ Xem chi tiết kỹ thuật &amp; Nhật ký huấn luyện CTC (JSON)</span><span>▾</span></summary>
        <pre id="sentenceTrainOutput">Chưa có nhật ký huấn luyện CTC.</pre>
      </details>
    </div>

    <!-- 2B. PREDICT SENTENCE -->
    <div class="card sentence-card">
      <div class="card-header">
        <h2><span class="step-badge">Bước 2</span> Giải mã Câu &amp; Dịch sang Tiếng Việt Tự nhiên</h2>
        <span style="font-size:0.78rem; color:var(--muted);">Word-Spotting Viterbi + CTC + Bigram LM</span>
      </div>

      <div class="hint-box">
        Hỗ trợ <strong>2 cơ chế nhận diện câu độc lập</strong> (chuyên trị chuyển ký hiệu đột ngột không hạ tay):<br/>
        • <strong>⚡ Phân đoạn Chuyển tiếp Đột ngột + Viterbi LM (Khuyên dùng):</strong> Bắt điểm gãy vận tốc cổ tay &amp; đổi hình dáng tay đột ngột giữa các ký hiệu liên tiếp mà không cần hạ tay reset, kết hợp Quy hoạch động Viterbi + Ngữ pháp Tiếng Việt!<br/>
        • <strong>🧠 CTC Beam Search + Ngữ pháp LM:</strong> Giải mã khung hình liên tục bằng mạng 2-Layer BiLSTM CTC kết hợp Bigram LM.
      </div>

      <div class="field">
        <label>Cơ chế giải mã câu ( áp dụng cho cả File <code>.npy</code> và Camera ):</label>
        <select id="sentEngineSelect">
          <option value="word_spotting">⚡ Chế độ 1 (Khuyên dùng): Phân đoạn Chuyển tiếp Đột ngột (Không cần hạ tay) + Viterbi Ngữ pháp LM</option>
          <option value="beam_search_lm">🧠 Chế độ 2: Mạng CTC Prefix Beam Search + Ngữ pháp Bigram LM</option>
          <option value="beam_search">🎯 Mạng CTC Prefix Beam Search thuần</option>
          <option value="greedy">⚡ Mạng CTC Greedy thuần</option>
        </select>
      </div>

      <div class="field">
        <label>Chọn 1 hoặc nhiều file <code>.npy</code> từ đơn để nối thành câu liên tục:</label>
        <div class="row">
          <input type="file" id="sentNpyFilesInput" accept=".npy" multiple style="flex:1;" />
          <button class="purple" onclick="predictSentenceFromUploadedNpy()">🔍 Giải mã Câu đã chọn</button>
        </div>
      </div>

      <div class="row">
        <button class="secondary" onclick="predictRandomTestSentenceNpy()" style="width:100%;">
          🎲 Tự động ghép 1 câu hợp ngữ pháp từ Tập Test (.npy) &amp; Giải mã ngay
        </button>
      </div>

      <div class="result-box">
        <div class="result-label">Kết quả Giải mã Câu &amp; Câu Tiếng Việt Tự nhiên</div>
        <div id="sentDecodedChips" style="color:var(--muted); font-size:0.9rem;">
          Hãy bấm <strong>"Tự động ghép 1 câu hợp ngữ pháp từ Tập Test"</strong> hoặc chọn nhiều file <code>.npy</code> ở trên.
        </div>
      </div>

      <details class="tech-details">
        <summary><span>📊 Xem chi tiết phân đoạn từng từ &amp; điểm số LM (JSON)</span><span>▾</span></summary>
        <pre id="sentencePredictOutput">Chưa có dữ liệu giải mã câu.</pre>
      </details>
    </div>

    <!-- 2C. TOKEN CONFUSION MATRIX VISUALIZATION FOR SENTENCE CTC -->
    <div class="card sentence-card" id="sentCmCard" style="grid-column: 1 / -1; display:none;">
      <div class="card-header" style="flex-wrap:wrap;">
        <h2 style="color:var(--purple-light);"><span class="step-badge">Đánh giá</span> Ma trận nhầm lẫn Từ vựng (Token Confusion Matrix) — Mô hình Câu CTC</h2>
        <div class="row">
          <button class="secondary" onclick="refreshSentenceConfusionMatrix(true)">🔄 Tính toán lại trên Tập Test</button>
          <button id="btnToggleSentNorm" class="secondary" onclick="toggleSentCmNormalize()">Chế độ: Tỉ lệ chuẩn hoá %</button>
          <a id="btnDownloadSentCm" class="btn-link secondary" download="vsl_sentence_confusion_matrix.png">⬇️ Tải ảnh PNG</a>
        </div>
      </div>
      <div id="sentCmSummaryText" style="font-size:0.92rem; color:var(--text-soft);">Đang tải ma trận nhầm lẫn...</div>
      <div class="cm-layout">
        <div style="min-width:0;">
          <div style="font-size:0.82rem; color:var(--muted); margin-bottom:6px;">
            Bảng nhiệt tương tác từ vựng — <em>Hàng ngang: Từ thực tế (True) &times; Cột dọc: Từ giải mã (Pred)</em>
          </div>
          <div id="sentCmTableContainer" style="overflow-x:auto; max-height:440px; overflow-y:auto; border:1px solid var(--border); border-radius:10px;"></div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:0.82rem; color:var(--muted); margin-bottom:6px;">Biểu đồ trực quan (Matplotlib Heatmap):</div>
          <img id="sentCmImg" style="width:100%; max-width:420px; border-radius:10px; border:1px solid var(--border); background:#060911;" alt="Sentence Confusion Matrix" />
        </div>
      </div>
    </div>
  </div>

  <!-- ===================================================================== -->
  <!-- UNIFIED HOLISTIC (76, 3) WEBCAM REALTIME & RECORDING PANEL            -->
  <!-- ===================================================================== -->
  <div class="container" style="padding-top: 0;">
    <div class="card" style="grid-column: 1 / -1;">
      <div class="card-header" style="flex-wrap:wrap;">
        <h2 id="webcamTitle"><span class="step-badge">Trực tiếp</span> Nhận diện qua Camera (MediaPipe Holistic 76 điểm)</h2>
        <div class="row">
          <button id="btnStartCam" onclick="startWebcam()">📷 1. Bật Camera</button>
          <button id="btnRecordClip" class="green" onclick="captureCleanClipAndPredict()" disabled>🔴 2. Quay 1 Ký hiệu (2.5s) &amp; Nhận diện</button>
          <button id="btnStopCam" class="secondary" onclick="stopWebcam()" disabled>⏹️ Tắt Camera</button>
          <button class="secondary" onclick="resetActiveStream()">🧹 Xoá Bộ đệm</button>
          <label style="display:inline-flex; align-items:center; gap:6px; color:var(--text-soft); font-size:0.82rem; cursor:pointer; background:var(--panel-alt); padding:7px 12px; border-radius:8px; border:1px solid var(--border);">
            <input type="checkbox" id="chkMirrorFlip" />
            Lật gương (Mirror)
          </label>
        </div>
      </div>

      <div class="webcam-grid">
        <div>
          <div class="video-wrap">
            <video id="webcamVideo" playsinline></video>
            <canvas id="webcamCanvas" width="640" height="480"></canvas>
            <div class="cam-overlay-bar">
              <div style="display:flex; gap:6px;">
                <span id="pillLeftHand" class="hand-pill">✋ Tay Trái (L)</span>
                <span id="pillRightHand" class="hand-pill">🤚 Tay Phải (R)</span>
              </div>
              <span id="camStatusText" class="hand-pill" style="color:var(--accent);">Camera đang tắt</span>
            </div>
          </div>
          <div style="font-size:0.78rem; color:var(--muted); margin-top:8px;">
            💡 <strong>Mẹo:</strong> Ngồi cách camera khoảng 0.8–1.2m sao cho khung hình thấy rõ 2 vai và 2 bàn tay. Để tắt "Lật gương" để khớp với góc quay gốc của bộ dữ liệu.
          </div>
        </div>

        <div style="display:flex; flex-direction:column; gap:12px;">
          <div class="result-box">
            <div class="result-label">Trạng thái Luồng Nhận diện</div>
            <div id="livePhaseText" style="font-size:0.98rem; font-weight:700; color:var(--accent);">Chưa bật Camera</div>
          </div>

          <div id="liveOutputBox" class="result-box" style="min-height:110px;">
            <div class="result-label">Kết quả Nhận diện từ Camera</div>
            <div id="liveResultContent" style="margin-top:6px; color:var(--muted);">
              Bật Camera và thực hiện ký hiệu hoặc bấm nút <strong>"Quay 1 Ký hiệu (2.5s)"</strong> để nhận kết quả chính xác nhất.
            </div>
          </div>

          <div class="result-box" style="border-color:rgba(168, 85, 247, 0.35);">
            <div style="font-size:0.86rem; color:var(--purple-light); font-weight:700; margin-bottom:4px;">
              ✨ Dạy Trực Tiếp &amp; Tinh chỉnh Góc máy Cá nhân (Few-Shot + Đa cửa sổ)
            </div>
            <div style="font-size:0.79rem; color:var(--muted); margin-bottom:10px;">
              Chọn từ bên dưới rồi bấm <strong>"🎯 Quay &amp; Dạy trực tiếp (2.5s)"</strong> để hệ thống tự đếm ngược, ghi hình động tác của bạn, tự cắt bỏ phần hạ tay thừa và cập nhật cả Mô hình Từ đơn lẫn Mô hình Câu + Bộ nhớ Prototype:
            </div>
            <div class="row">
              <select id="webcamAdaptLabelSelect" style="flex:1; min-width:160px;"></select>
              <button id="btnRecordAndTeach" class="purple" onclick="recordAndTeachDirectly()">🎯 Quay &amp; Dạy trực tiếp (2.5s)</button>
              <button id="btnFinetuneCam" class="secondary" onclick="finetuneWithLastWebcamClip()" title="Dạy bằng clip vừa quay ở nút số 2 hoặc bộ đệm hiện tại">⚡ Dạy từ Clip vừa quay</button>
              <button id="btnClearWebcamLabel" class="secondary" onclick="clearWebcamSamplesForLabel()" title="Xoá các mẫu webcam cũ bị lỗi của từ này" style="padding:8px 11px;">🗑️ Xoá mẫu cũ</button>
            </div>
            <div id="webcamAdaptStatus" style="font-size:0.82rem; color:var(--success); margin-top:6px; font-weight:500;"></div>
          </div>

          <details id="telemetryDetails" class="tech-details">
            <summary><span>📡 Xem luồng dữ liệu Telemetry trực tiếp (JSON)</span><span>▾</span></summary>
            <pre id="liveTelemetry" style="max-height:130px;">Telemetry trực tiếp.</pre>
          </details>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentMode = 'word';
    window.latestWordClasses = [];
    window.latestSentenceClasses = [];
    window.webcamSampleCounts = {};

    function updateWebcamAdaptDropdown() {
      const sel = document.getElementById('webcamAdaptLabelSelect');
      if (!sel) return;
      const list = currentMode === 'word' ? (window.latestWordClasses || []) : (window.latestSentenceClasses || []);
      const curVal = sel.value;
      const counts = window.webcamSampleCounts || {};
      sel.innerHTML = list.map(c => {
        const cnt = counts[c] || 0;
        const suffix = cnt > 0 ? ` (Đã dạy: ${cnt} mẫu)` : '';
        return `<option value="${c}">${c}${suffix}</option>`;
      }).join('');
      if (curVal && list.includes(curVal)) sel.value = curVal;
    }

    function switchMode(mode) {
      currentMode = mode;
      pendingWordFrames = [];
      pendingSentenceFrames = [];
      document.getElementById('panelWord').classList.toggle('hidden', mode !== 'word');
      document.getElementById('panelSentence').classList.toggle('hidden', mode !== 'sentence');
      document.getElementById('tabBtnWord').className = mode === 'word' ? 'tab-btn active-word' : 'tab-btn';
      document.getElementById('tabBtnSentence').className = mode === 'sentence' ? 'tab-btn active-sentence' : 'tab-btn';
      const btnRec = document.getElementById('btnRecordClip');
      if (btnRec) {
        btnRec.innerHTML = mode === 'word'
          ? '🔴 2. Quay 1 Ký hiệu Từ đơn (2.4s) &amp; Nhận diện'
          : '🔴 2. Quay 1 Câu Liên tục (4.0s, không cần hạ tay) &amp; Giải mã';
      }
      updateWebcamAdaptDropdown();
    }

    function arrayBufferToBase64(buffer) {
      let binary = '';
      const bytes = new Uint8Array(buffer);
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
      }
      return btoa(binary);
    }

    async function refreshStatus() {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.default_npy_root) {
        document.getElementById('npyRootInput').value = data.default_npy_root;
      }
      window.latestWordClasses = data.word_classes || [];
      window.latestSentenceClasses = data.sentence_classes || [];
      window.webcamSampleCounts = data.webcam_sample_counts || {};
      const wordPill = data.word_model_loaded
        ? `<span class="status-pill" title="${window.latestWordClasses.join(', ')}"><span class="status-dot online"></span> Phần 1 (Từ đơn): <strong>${window.latestWordClasses.length} từ riêng</strong></span>`
        : `<span class="status-pill"><span class="status-dot"></span> Phần 1 (Từ đơn): Chưa nạp</span>`;
      const sentPill = data.sentence_model_loaded
        ? `<span class="status-pill" title="${window.latestSentenceClasses.join(', ')}"><span class="status-dot online"></span> Phần 2 (Câu liên tục): <strong>${window.latestSentenceClasses.length} từ riêng (Độc lập)</strong></span>`
        : `<span class="status-pill"><span class="status-dot"></span> Phần 2 (Câu liên tục): Chưa nạp</span>`;
      document.getElementById('headerStats').innerHTML = wordPill + sentPill;

      updateWebcamAdaptDropdown();

      const gpBox = document.getElementById('grammarPreviewBox');
      if (gpBox && data.grammar_examples && data.grammar_examples.length > 0) {
        const tplNames = (data.grammar_templates || []).map(t => `<code style="background:#fff; padding:1px 5px; border-radius:4px; border:1px solid #ddd6fe; font-size:0.75rem;">${t.name}: ${t.slots.join(' + ')}</code>`).join(' ');
        const exList = data.grammar_examples.slice(0, 5).map(ex =>
          `<div style="margin-top:3px;">• <strong>${ex.glosses.join(' &rarr; ')}</strong> &rArr; <span style="color:#047857; font-weight:700;">"${ex.vietnamese}"</span></div>`
        ).join('');
        gpBox.innerHTML = `
          <div>🔒 <strong>Bộ từ vựng riêng của Mô hình Câu (${window.latestSentenceClasses.length} từ, tách biệt hoàn toàn với Phần 1):</strong> <code style="color:#6d28d9;">${window.latestSentenceClasses.join(', ')}</code></div>
          <div style="margin-top:5px;">✨ <strong>Đã kích hoạt ${(data.grammar_templates || []).length} cấu trúc câu tiếng Việt:</strong></div>
          <div style="margin-top:4px; display:flex; flex-wrap:wrap; gap:4px;">${tplNames}</div>
          <div style="margin-top:6px; border-top:1px dashed #c4b5fd; padding-top:4px;"><strong>Ví dụ câu tự động biến tấu &amp; dịch nghĩa:</strong>${exList}</div>
        `;
      }

      if (data.word_has_cm && !currentWordCmData) {
        refreshWordConfusionMatrix(false);
      }
      if (data.sentence_has_cm && !currentSentCmData) {
        refreshSentenceConfusionMatrix(false);
      }
    }

    let currentWordCmData = null;
    let wordCmIsNormalized = false;
    let currentSentCmData = null;
    let sentCmIsNormalized = false;

    function renderConfusionMatrixGrid(classNames, matrix, normMatrix, isNorm, themeColor = 'emerald') {
      if (!matrix || !matrix.length || !classNames || !classNames.length) {
        return '<div style="padding:16px; color:var(--muted); text-align:center;">Chưa có dữ liệu ma trận nhầm lẫn.</div>';
      }
      const n = classNames.length;
      let html = '<table class="cm-table"><thead><tr>';
      html += '<th class="cm-row-header" style="z-index:4;">Thực tế \\ Dự đoán</th>';
      for (let j = 0; j < n; j++) {
        html += `<th title="Dự đoán: ${classNames[j]}">${classNames[j]}</th>`;
      }
      html += '<th>Tổng</th><th>Recall</th></tr></thead><tbody>';

      for (let i = 0; i < n; i++) {
        const rowSum = matrix[i].reduce((a, b) => a + b, 0);
        const correct = matrix[i][i] || 0;
        const recall = rowSum > 0 ? (correct / rowSum) : 0;
        html += `<tr><th class="cm-row-header" title="Thực tế: ${classNames[i]}">${classNames[i]}</th>`;

        for (let j = 0; j < n; j++) {
          const val = matrix[i][j];
          const pct = normMatrix && normMatrix[i] ? (normMatrix[i][j] * 100) : (rowSum > 0 ? (val / rowSum * 100) : 0);
          const isDiag = (i === j);

          let bg = '#020617';
          let textColor = '#cbd5e1';
          if (isDiag) {
            const alpha = 0.18 + 0.82 * Math.min(1.0, (pct / 100));
            bg = themeColor === 'purple' ? `rgba(168, 85, 247, ${alpha})` : `rgba(16, 185, 129, ${alpha})`;
            if (pct > 50) textColor = '#ffffff';
          } else if (val > 0) {
            const errAlpha = 0.2 + 0.8 * Math.min(1.0, (pct / 50));
            bg = `rgba(244, 63, 94, ${errAlpha})`;
            if (pct > 20) textColor = '#ffffff';
          }

          const displayTxt = isNorm ? `${pct.toFixed(1)}%` : val;
          const tooltip = `Thực tế: ${classNames[i]} &#10;Dự đoán: ${classNames[j]} &#10;Số lượng: ${val} mẫu (${pct.toFixed(1)}%)`;

          html += `<td class="cm-cell" style="background:${bg}; color:${textColor}; font-weight:${isDiag ? 'bold' : 'normal'};" title="${tooltip}">${displayTxt}</td>`;
        }

        html += `<td style="font-weight:600; color:#94a3b8;">${rowSum}</td>`;
        html += `<td style="font-weight:bold; color:${recall >= 0.8 ? '#22c55e' : (recall >= 0.5 ? '#f59e0b' : '#ef4444')};">${(recall * 100).toFixed(1)}%</td>`;
        html += '</tr>';
      }

      // Summary Precision Row
      html += '<tr><th class="cm-row-header" style="color:#38bdf8;">Precision</th>';
      for (let j = 0; j < n; j++) {
        let colSum = 0;
        for (let i = 0; i < n; i++) colSum += matrix[i][j];
        const correct = matrix[j][j] || 0;
        const prec = colSum > 0 ? (correct / colSum) : 0;
        html += `<td style="font-weight:bold; color:${prec >= 0.8 ? '#22c55e' : (prec >= 0.5 ? '#f59e0b' : '#ef4444')};">${(prec * 100).toFixed(1)}%</td>`;
      }
      html += '<td>—</td><td>—</td></tr>';

      html += '</tbody></table>';
      return html;
    }

    function renderPerClassMetricsTable(classNames, perClassMetrics) {
      if (!perClassMetrics || Object.keys(perClassMetrics).length === 0) return '';
      let html = '<table class="cm-metrics-table"><thead><tr>';
      html += '<th>Lớp Từ (Class)</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Số mẫu Test (Support)</th></tr></thead><tbody>';

      for (const name of classNames) {
        const m = perClassMetrics[name];
        if (!m) continue;
        const p = ((m.precision || 0) * 100).toFixed(1);
        const r = ((m.recall || 0) * 100).toFixed(1);
        const f = ((m.f1 || 0) * 100).toFixed(1);
        const s = m.support || 0;
        html += `<tr>
          <td style="font-weight:600; color:#f8fafc;">${name}</td>
          <td><span style="color:#38bdf8; font-weight:600;">${p}%</span></td>
          <td><span style="color:#22c55e; font-weight:600;">${r}%</span></td>
          <td><span style="color:#c084fc; font-weight:bold;">${f}%</span></td>
          <td style="color:#94a3b8;">${s}</td>
        </tr>`;
      }
      html += '</tbody></table>';
      return html;
    }

    function renderWordConfusionMatrix(data) {
      if (!data) return;
      currentWordCmData = data;
      const card = document.getElementById('wordCmCard');
      if (card) card.style.display = 'block';

      const cm = data.confusion_matrix;
      const classes = data.class_names || [];
      const norm = data.normalized_matrix;
      const img = data.confusion_matrix_image || data.image_base64;

      const acc = (data.overall_accuracy !== undefined ? data.overall_accuracy : ((data.accuracy || 0) * 100)).toFixed(2);
      const macroF1 = data.macro_f1 !== undefined ? (data.macro_f1 * 100).toFixed(1) : null;
      const total = data.total_samples || (cm ? cm.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0) : 0);

      document.getElementById('wordCmSummaryText').innerHTML =
        `Độ chính xác tổng thể tập Test (Accuracy): <strong style="color:#22c55e; font-size:1.05rem;">${acc}%</strong> | Tổng số mẫu kiểm thử: <strong>${total}</strong> ${macroF1 ? `| Macro-F1: <strong style="color:#c084fc;">${macroF1}%</strong>` : ''}`;

      document.getElementById('wordCmTableContainer').innerHTML = renderConfusionMatrixGrid(classes, cm, norm, wordCmIsNormalized, 'emerald');

      if (data.per_class_metrics) {
        document.getElementById('wordCmMetricsContainer').innerHTML = renderPerClassMetricsTable(classes, data.per_class_metrics);
      } else {
        document.getElementById('wordCmMetricsContainer').innerHTML = '';
      }

      const imgEl = document.getElementById('wordCmImg');
      const dlBtn = document.getElementById('btnDownloadWordCm');
      if (img && imgEl) {
        imgEl.src = img;
        imgEl.style.display = 'block';
        if (dlBtn) dlBtn.href = img;
      }
    }

    function toggleWordCmNormalize() {
      wordCmIsNormalized = !wordCmIsNormalized;
      document.getElementById('btnToggleWordNorm').textContent = wordCmIsNormalized ? 'Chế độ: Số lượng mẫu' : 'Chế độ: Tỉ lệ chuẩn hoá %';
      if (currentWordCmData) {
        renderWordConfusionMatrix(currentWordCmData);
      }
    }

    async function refreshWordConfusionMatrix(forceReeval = false) {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      document.getElementById('wordCmSummaryText').textContent = 'Đang tính toán ma trận nhầm lẫn từ tập Test...';
      try {
        const res = await fetch('/api/word/confusion_matrix', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ root_dir })
        });
        const json = await res.json();
        if (json.ok && json.data) {
          renderWordConfusionMatrix(json.data);
        } else {
          document.getElementById('wordCmSummaryText').textContent = `Lỗi: ${json.error}`;
        }
      } catch (err) {
        document.getElementById('wordCmSummaryText').textContent = `Lỗi kết nối: ${err.message}`;
      }
    }

    function renderSentenceConfusionMatrix(data) {
      if (!data) return;
      currentSentCmData = data;
      const card = document.getElementById('sentCmCard');
      if (card) card.style.display = 'block';

      const cm = data.confusion_matrix;
      const classes = data.class_names || [];
      const norm = data.normalized_matrix;
      const img = data.confusion_matrix_image || data.image_base64;

      const acc = (data.overall_accuracy !== undefined ? data.overall_accuracy : ((data.token_accuracy || 0) * 100)).toFixed(2);
      const total = data.total_samples || data.total_tokens || (cm ? cm.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0) : 0);

      document.getElementById('sentCmSummaryText').innerHTML =
        `Độ chính xác Token Gloss (Token Accuracy): <strong style="color:#c084fc; font-size:1.05rem;">${acc}%</strong> | Tổng số token đối sánh: <strong>${total}</strong>`;

      document.getElementById('sentCmTableContainer').innerHTML = renderConfusionMatrixGrid(classes, cm, norm, sentCmIsNormalized, 'purple');

      const imgEl = document.getElementById('sentCmImg');
      const dlBtn = document.getElementById('btnDownloadSentCm');
      if (img && imgEl) {
        imgEl.src = img;
        imgEl.style.display = 'block';
        if (dlBtn) dlBtn.href = img;
      }
    }

    function toggleSentCmNormalize() {
      sentCmIsNormalized = !sentCmIsNormalized;
      document.getElementById('btnToggleSentNorm').textContent = sentCmIsNormalized ? 'Chế độ: Số lượng mẫu' : 'Chế độ: Tỉ lệ chuẩn hoá %';
      if (currentSentCmData) {
        renderSentenceConfusionMatrix(currentSentCmData);
      }
    }

    async function refreshSentenceConfusionMatrix(forceReeval = false) {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      document.getElementById('sentCmSummaryText').textContent = 'Đang tính toán ma trận nhầm lẫn Gloss từ tập Test...';
      try {
        const res = await fetch('/api/sentence/confusion_matrix', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ root_dir })
        });
        const json = await res.json();
        if (json.ok && json.data) {
          renderSentenceConfusionMatrix(json.data);
        } else {
          document.getElementById('sentCmSummaryText').textContent = `Lỗi: ${json.error}`;
        }
      } catch (err) {
        document.getElementById('sentCmSummaryText').textContent = `Lỗi kết nối: ${err.message}`;
      }
    }

    async function scanNpyFolder() {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      const res = await fetch('/api/npy/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ root_dir })
      });
      const data = await res.json();
      if (data.ok) {
        document.getElementById('npyScanSummary').textContent =
          `Tìm thấy ${data.info.total_classes} lớp từ! Các lớp giàu dữ liệu mặc định: ${(data.info.recommended_folders || []).slice(0, 8).join(', ')}`;
      } else {
        document.getElementById('npyScanSummary').textContent = `Lỗi: ${data.error}`;
      }
    }

    function parseSelectedFolders(inputId) {
      const raw = document.getElementById(inputId).value.trim();
      if (!raw) return null;
      return raw.split(',').map(s => s.trim()).filter(Boolean);
    }

    // ---------------- PART 1: WORD (BiLSTM) ----------------
    async function trainWordFromNpy() {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      const selected_folders = parseSelectedFolders('wordSelectedFolders');
      const max_classes = parseInt(document.getElementById('wordMaxClasses').value || '8', 10);
      const max_files = parseInt(document.getElementById('wordMaxFiles').value || '35', 10);
      const epochs = parseInt(document.getElementById('wordEpochs').value || '35', 10);
      const batch_size = parseInt(document.getElementById('wordBatch').value || '8', 10);
      const lr = parseFloat(document.getElementById('wordLr').value || '0.0015');

      const btn = document.getElementById('btnTrainWordNpy');
      btn.disabled = true;
      document.getElementById('wordTrainMsg').textContent =
        'Đang đọc các file .npy từ keypoints_splited, nội suy khuyết tay, cân bằng lớp (30x138) và huấn luyện BiLSTM...';

      try {
        const res = await fetch('/api/word/train_npy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            root_dir,
            selected_folders,
            max_classes,
            max_files_per_class: max_files,
            epochs,
            batch_size,
            lr
          })
        });
        const data = await res.json();
        document.getElementById('wordTrainOutput').textContent = JSON.stringify(data, null, 2);
        if (data.ok) {
          const valAcc = (data.summary.best_val_accuracy * 100).toFixed(1);
          const testAcc = (data.summary.test_metrics.accuracy * 100).toFixed(1);
          const macroF1 = (data.summary.test_metrics.macro_f1 * 100).toFixed(1);
          document.getElementById('wordTrainMsg').textContent =
            `Huấn luyện BiLSTM hoàn tất (${data.summary.class_names.join(', ')})! Val Acc = ${valAcc}% | Test Acc = ${testAcc}% | Macro-F1 = ${macroF1}%`;
          if (data.summary && data.summary.test_metrics) {
            renderWordConfusionMatrix(data.summary.test_metrics);
          }
          await refreshStatus();
        } else {
          document.getElementById('wordTrainMsg').textContent = `Lỗi: ${data.error}`;
        }
      } finally {
        btn.disabled = false;
      }
    }

    function renderWordBadge(res, groundTruth) {
      if (!res) return '—';
      const color = res.status === 'ACCEPTED' ? '#10b981' : (res.status === 'AMBIGUOUS' ? '#f59e0b' : '#ef4444');
      const statusText = res.status === 'ACCEPTED' ? 'Độ tin cậy cao' : (res.status === 'AMBIGUOUS' ? 'Cần thực hiện rõ hơn' : 'Chưa rõ ký hiệu');
      const gtHtml = groundTruth
        ? `<div style="margin-bottom:8px; font-size:0.82rem; color:var(--accent);">📌 Nguồn kiểm thử: <strong>${groundTruth}</strong></div>`
        : '';
      const top3Bars = (res.top3 || []).map(t => {
        const pct = Math.max(0, Math.min(100, t.prob * 100));
        return `
          <div class="top3-item">
            <span style="font-weight:600; color:var(--text-soft); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${t.label}">${t.label}</span>
            <div class="top3-bar-bg"><div class="top3-bar-fill" style="width:${pct.toFixed(1)}%;"></div></div>
            <span style="text-align:right; font-weight:700; color:var(--accent);">${pct.toFixed(1)}%</span>
          </div>
        `;
      }).join('');
      return `
        ${gtHtml}
        <div class="pred-hero">
          <div>
            <span class="word-badge" style="background:${color};">${res.predicted_label}</span>
          </div>
          <div style="font-size:0.84rem; color:var(--text-soft); text-align:right;">
            <div>Từ khớp nhất: <strong style="color:#fff;">${res.raw_top1_label}</strong> (${(res.confidence * 100).toFixed(1)}%)</div>
            <div style="font-size:0.77rem; color:${color}; font-weight:600; margin-top:2px;">● ${statusText}</div>
          </div>
        </div>
        ${top3Bars ? `<div class="top3-list">${top3Bars}</div>` : ''}
      `;
    }

    async function predictWordFromUploadedNpy() {
      const fInput = document.getElementById('wordNpyFileInput');
      if (!fInput.files || fInput.files.length === 0) {
        alert('Hãy chọn 1 file .npy trên máy tính của bạn.');
        return;
      }
      const file = fInput.files[0];
      const b64 = arrayBufferToBase64(await file.arrayBuffer());
      const res = await fetch('/api/word/predict_npy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name, npy_base64: b64 })
      });
      const data = await res.json();
      document.getElementById('wordPredictOutput').textContent = JSON.stringify(data, null, 2);
      if (data.ok) {
        document.getElementById('wordSampleBadge').innerHTML = renderWordBadge(data.result, data.filename);
      }
    }

    async function predictRandomTestNpyWord() {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      const res = await fetch('/api/word/predict_random_test_npy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ root_dir })
      });
      const data = await res.json();
      document.getElementById('wordPredictOutput').textContent = JSON.stringify(data, null, 2);
      if (data.ok) {
        document.getElementById('wordSampleBadge').innerHTML =
          renderWordBadge(data.result, `${data.true_label} (${data.file_path})`);
      }
    }

    // ---------------- PART 2: SENTENCE (2-LAYER BiLSTM + CTC + GRAMMAR LM + WORD SPOTTING) ----------------
    function syncFoldersFromWordModel() {
      if (currentWordClasses && currentWordClasses.length > 0) {
        document.getElementById('sentSelectedFolders').value = currentWordClasses.join(', ');
        document.getElementById('sentMaxClasses').value = currentWordClasses.length;
      }
    }

    async function trainSentenceFromNpy() {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      const selected_folders = parseSelectedFolders('sentSelectedFolders');
      const max_classes = parseInt(document.getElementById('sentMaxClasses').value || '12', 10);
      const sentences_per_split = parseInt(document.getElementById('sentCount').value || '150', 10);
      const epochs = parseInt(document.getElementById('sentenceEpochs').value || '30', 10);
      const batch_size = parseInt(document.getElementById('sentenceBatch').value || '8', 10);
      const use_grammar = document.getElementById('chkUseSentenceGrammar') ? document.getElementById('chkUseSentenceGrammar').checked : true;

      const btn = document.getElementById('btnTrainSentNpy');
      btn.disabled = true;
      document.getElementById('sentenceTrainMsg').textContent =
        '⏳ Đang biến tấu câu theo Ngữ pháp VSL từ từ đơn (.npy) và huấn luyện mạng 2-Layer BiLSTM + CTC...';

      try {
        const res = await fetch('/api/sentence/train_npy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            root_dir,
            selected_folders,
            max_classes,
            sentences_per_split,
            epochs,
            batch_size,
            use_grammar
          })
        });
        const data = await res.json();
        document.getElementById('sentenceTrainOutput').textContent = JSON.stringify(data, null, 2);
        if (data.ok) {
          const valWer = (data.summary.best_val_wer * 100).toFixed(1);
          const testWer = (data.summary.test_metrics.corpus_wer * 100).toFixed(1);
          document.getElementById('sentenceTrainMsg').textContent =
            `✅ Huấn luyện CTC hoàn tất! Tỉ lệ lỗi từ (Test WER) = ${testWer}% | Khớp chính xác cả câu = ${(data.summary.test_metrics.exact_match_rate * 100).toFixed(1)}%`;
          if (data.summary && data.summary.test_metrics) {
            renderSentenceConfusionMatrix(data.summary.test_metrics);
          }
          await refreshStatus();
        } else {
          document.getElementById('sentenceTrainMsg').textContent = `❌ Lỗi: ${data.error}`;
        }
      } finally {
        btn.disabled = false;
      }
    }

    function renderSentenceChips(decodedGlosses, trueGlosses, extra = null) {
      const engineLabels = {
        word_spotting: '🎯 Zero-Shot Word-Spotting (BiLSTM Từ Đơn + Viterbi DP + Ngữ Pháp)',
        beam_search_lm: '🧠 CTC Beam Search + Mô hình Ngôn ngữ Ngữ pháp (LM)',
        beam_search: '🔍 CTC Prefix Beam Search',
        greedy: '⚡ CTC Greedy Decode'
      };
      const engineBadge = (extra && extra.engine)
        ? `<div style="margin-bottom:6px; font-size:0.76rem; color:#6d28d9; font-weight:700;">Chế độ giải mã: ${engineLabels[extra.engine] || extra.engine}</div>`
        : '';

      const gtVie = (extra && extra.true_vietnamese) ? ` <span style="color:#047857;">("${extra.true_vietnamese}")</span>` : '';
      const gtHtml = trueGlosses
        ? `<div style="margin-bottom:8px; font-size:0.83rem; color:var(--accent);">📌 Câu gốc (Nhãn thực tế): <strong>${trueGlosses.join(' &rarr; ')}</strong>${gtVie}</div>`
        : '';

      const chips = (decodedGlosses && decodedGlosses.length > 0)
        ? decodedGlosses.map((g, idx) => {
            const seg = (extra && extra.segments && extra.segments[idx]) ? extra.segments[idx] : null;
            const segMeta = seg
              ? `<span style="font-size:0.7rem; opacity:0.85; margin-left:5px; font-weight:500;">[${seg.start_frame}..${seg.end_frame}] ${(seg.confidence * 100).toFixed(0)}%</span>`
              : '';
            return `<span class="gloss-chip">${g}${segMeta}</span>`;
          }).join(' <span style="color:var(--muted); font-weight:700;">&rarr;</span> ')
        : '<em style="color:var(--muted);">Chưa phát hiện từ nào trong chuỗi</em>';

      const vieSentence = (extra && extra.vietnamese_sentence) ? extra.vietnamese_sentence : '';
      const vieHtml = vieSentence
        ? `<div style="margin-top:10px; padding:9px 12px; border-radius:8px; background:linear-gradient(135deg, #ecfdf5, #f0fdf4); border:1px solid #6ee7b7; color:#065f46; font-size:0.95rem;">
             💬 <strong>Câu tiếng Việt tự nhiên:</strong> <span style="font-size:1.05rem; font-weight:800; color:#047857;">"${vieSentence}"</span>
           </div>`
        : '';

      return `${engineBadge}${gtHtml}<div style="display:flex; align-items:center; flex-wrap:wrap; gap:4px;">${chips}</div>${vieHtml}`;
    }

    async function predictSentenceFromUploadedNpy(decodeMethod) {
      const fInput = document.getElementById('sentNpyFilesInput');
      if (!fInput.files || fInput.files.length === 0) {
        alert('Hãy chọn 1 hoặc nhiều file .npy trên máy tính để ghép thành câu và giải mã.');
        return;
      }
      const chosenEngine = decodeMethod || (document.getElementById('sentEngineSelect') ? document.getElementById('sentEngineSelect').value : 'word_spotting');
      const filesPayload = [];
      for (const f of fInput.files) {
        filesPayload.push({
          filename: f.name,
          npy_base64: arrayBufferToBase64(await f.arrayBuffer())
        });
      }
      const res = await fetch('/api/sentence/predict_npy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decode_method: chosenEngine, files: filesPayload })
      });
      const data = await res.json();
      document.getElementById('sentencePredictOutput').textContent = JSON.stringify(data, null, 2);
      if (data.ok) {
        document.getElementById('sentDecodedChips').innerHTML =
          renderSentenceChips(data.result.predicted_glosses, null, {
            vietnamese_sentence: data.result.vietnamese_sentence,
            segments: data.result.segments,
            engine: data.decode_method || chosenEngine
          });
      }
    }

    async function predictRandomTestSentenceNpy() {
      const root_dir = document.getElementById('npyRootInput').value.trim();
      const chosenEngine = document.getElementById('sentEngineSelect') ? document.getElementById('sentEngineSelect').value : 'word_spotting';
      const res = await fetch('/api/sentence/predict_random_test_npy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ root_dir, decode_method: chosenEngine, use_grammar: true })
      });
      const data = await res.json();
      document.getElementById('sentencePredictOutput').textContent = JSON.stringify(data, null, 2);
      if (data.ok) {
        const resObj = data.result || data.greedy_result || {};
        document.getElementById('sentDecodedChips').innerHTML =
          renderSentenceChips(resObj.predicted_glosses, data.true_glosses, {
            vietnamese_sentence: resObj.vietnamese_sentence,
            true_vietnamese: data.true_vietnamese,
            segments: resObj.segments,
            engine: data.decode_method || chosenEngine
          });
      }
    }

    // ---------------- WEBCAM (HOLISTIC 76x3 + BLAZEPALM HANDS FUSION) ----------------
    let cameraObj = null;
    let holisticObj = null;
    let handsObj = null;
    let latestHandsResult = { multiHandLandmarks: [], multiHandedness: [], ts: 0 };
    const prevValidWrist = { L: null, R: null };
    const stableShoulders = { p11: null, p12: null, width: 0.28 };
    let isSendingFrame = false;
    let pendingSentenceFrames = [];
    let pendingWordFrames = [];
    let isRecordingClip = false;
    let recordedClip76x3 = [];
    let lastCapturedClip76x3 = [];

    // Pose pairs to swap when mirrorFlip is enabled: (1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)
    const POSE_LR_PAIRS = [
      [1,4],[2,5],[3,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,22],[23,24],[25,26],[27,28],[29,30],[31,32]
    ];

    // Real camera aspect (width/height). VSL400 training videos are 224x224, so the server
    // rescales webcam x-coordinates by this factor to undo the 4:3 / 16:9 anisotropy.
    function camAspect() {
      const v = document.getElementById('webcamVideo');
      return (v && v.videoWidth > 0 && v.videoHeight > 0) ? (v.videoWidth / v.videoHeight) : (4 / 3);
    }

    // Validate 21-point hand skeleton geometry so Holistic's crumpled/folded-back ROI crops
    // (when the right hand is held horizontally across the chest/shoulder) are detected & rejected.
    function isHandGeometryValid(lm21) {
      if (!lm21 || lm21.length < 21) return false;
      let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
      for (let i = 0; i < 21; i++) {
        const p = lm21[i];
        if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) return false;
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }
      const diag = Math.hypot(maxX - minX, maxY - minY);
      if (diag < 0.032 || diag > 0.68) return false;

      const palmLen = Math.hypot(lm21[9].x - lm21[0].x, lm21[9].y - lm21[0].y);
      if (palmLen < 0.020 || palmLen > 0.36) return false;

      // Knuckle arch width (Index MCP 5 to Pinky MCP 17) must not collapse to a single point
      const knuckleSpan = Math.hypot(lm21[5].x - lm21[17].x, lm21[5].y - lm21[17].y);
      if (knuckleSpan < 0.16 * palmLen || knuckleSpan > 1.85 * palmLen) return false;

      // Unit vector along palm axis (Wrist 0 -> Middle MCP 9)
      const ux = (lm21[9].x - lm21[0].x) / palmLen;
      const uy = (lm21[9].y - lm21[0].y) / palmLen;

      // Check finger bones and ensure fingers aren't folded backward behind the wrist
      // (the exact Holistic ROI crop bug when the right hand is horizontal in front of the chest)
      const fingers = [[5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]];
      let backwardTips = 0;
      let foldedBackFingers = 0;
      for (const [mcp, pip, dip, tip] of fingers) {
        const mcpProj = (lm21[mcp].x - lm21[0].x) * ux + (lm21[mcp].y - lm21[0].y) * uy;
        if (mcpProj < 0.25 * palmLen) return false;

        const b1 = Math.hypot(lm21[pip].x - lm21[mcp].x, lm21[pip].y - lm21[mcp].y);
        const b2 = Math.hypot(lm21[dip].x - lm21[pip].x, lm21[dip].y - lm21[pip].y);
        const b3 = Math.hypot(lm21[tip].x - lm21[dip].x, lm21[tip].y - lm21[dip].y);
        if (b1 > 1.35 * palmLen || b2 > 1.15 * palmLen || b3 > 1.05 * palmLen) return false;

        const tipProj = (lm21[tip].x - lm21[0].x) * ux + (lm21[tip].y - lm21[0].y) * uy;
        if (tipProj < -0.08 * palmLen) backwardTips += 1;

        // If PIP->TIP points sharply backward against Wrist->MCP while still long, the crop box cut off the fingers
        const v1x = lm21[mcp].x - lm21[0].x, v1y = lm21[mcp].y - lm21[0].y;
        const v2x = lm21[tip].x - lm21[pip].x, v2y = lm21[tip].y - lm21[pip].y;
        const l1 = Math.hypot(v1x, v1y), l2 = Math.hypot(v2x, v2y);
        if (l1 > 1e-4 && l2 > 0.42 * palmLen) {
          const cosAng = (v1x * v2x + v1y * v2y) / (l1 * l2);
          if (cosAng < -0.65 && tipProj < 0.45 * palmLen) foldedBackFingers += 1;
        }
      }
      if (backwardTips >= 2 || foldedBackFingers >= 2) return false;
      return true;
    }

    // Fuse standalone @mediapipe/hands (BlazePalm detector) into Holistic results so horizontal /
    // chest-level hands never crumple from Pose wrist/index/pinky crop errors.
    function fuseBlazePalmHandsIntoHolistic(results) {
      const pose = (results.poseLandmarks && results.poseLandmarks.length >= 33) ? results.poseLandmarks : null;
      const p15 = pose ? pose[15] : null; // anatomical Left wrist (right side of raw un-mirrored image)
      const p16 = pose ? pose[16] : null; // anatomical Right wrist (left side of raw un-mirrored image)
      const bodyCenterX = (pose && pose[11] && pose[12])
        ? 0.5 * (pose[11].x + pose[12].x)
        : 0.5;

      // 1) Validate existing Holistic hands first; discard if crumpled/corrupted
      const holL = isHandGeometryValid(results.leftHandLandmarks) ? results.leftHandLandmarks : null;
      const holR = isHandGeometryValid(results.rightHandLandmarks) ? results.rightHandLandmarks : null;
      const rawHolLWrist = (results.leftHandLandmarks && results.leftHandLandmarks[0]) ? results.leftHandLandmarks[0] : null;
      const rawHolRWrist = (results.rightHandLandmarks && results.rightHandLandmarks[0]) ? results.rightHandLandmarks[0] : null;

      results.leftHandLandmarks = holL;
      results.rightHandLandmarks = holR;

      // 2) Incorporate fresh BlazePalm (@mediapipe/hands) detections
      const now = performance.now();
      const rawHands = (latestHandsResult && (now - latestHandsResult.ts) < 180)
        ? (latestHandsResult.multiHandLandmarks || [])
        : [];
      const rawHandedness = (latestHandsResult && latestHandsResult.multiHandedness) || [];

      const validCandidates = [];
      for (let i = 0; i < rawHands.length; i++) {
        const h = rawHands[i];
        if (!isHandGeometryValid(h)) continue;
        // Deduplicate if BlazePalm emitted two overlapping boxes for the same physical hand
        let dup = false;
        for (const existing of validCandidates) {
          if (Math.hypot(h[0].x - existing.lm[0].x, h[0].y - existing.lm[0].y) < 0.075) {
            dup = true;
            break;
          }
        }
        if (dup) continue;
        const hInfo = rawHandedness[i];
        const rawLabel = (hInfo && (hInfo.label || (hInfo.classification && hInfo.classification[0] && hInfo.classification[0].label))) || '';
        validCandidates.push({ lm: h, label: String(rawLabel) });
      }

      // Cost of assigning a BlazePalm hand candidate `c` to anatomical `side` ('L' or 'R')
      // Note: On un-mirrored videoElement, Holistic's `rightHandLandmarks` is anatomical Right ('R', near p16, x < center),
      // and MediaPipe Hands' default selfie-mode labels anatomical Right as 'Left'.
      function assignmentCost(c, side) {
        const w = c.lm[0];
        let cost = 0.0;
        const targetPoseW = (side === 'R') ? p16 : p15;
        const otherPoseW  = (side === 'R') ? p15 : p16;
        if (targetPoseW && (targetPoseW.visibility === undefined || targetPoseW.visibility > 0.3)) {
          const dTarget = Math.hypot(w.x - targetPoseW.x, w.y - targetPoseW.y);
          cost += 1.4 * Math.min(0.6, dTarget);
          // If only one arm is raised in Pose, strongly favor that raised arm
          if (otherPoseW && (otherPoseW.y - targetPoseW.y) > 0.16 && otherPoseW.y > 0.72) {
            cost -= 0.35;
          }
        }
        const rawHolW = (side === 'R') ? rawHolRWrist : rawHolLWrist;
        if (rawHolW && Math.hypot(w.x - rawHolW.x, w.y - rawHolW.y) < 0.15) {
          cost -= 0.30;
        }
        const prevW = prevValidWrist[side];
        if (prevW && (now - prevW.ts) < 220 && Math.hypot(w.x - prevW.x, w.y - prevW.y) < 0.15) {
          cost -= 0.25;
        }
        // Un-mirrored MediaPipe Hands label ('Left' = anatomical Right 'R', 'Right' = anatomical Left 'L')
        if (side === 'R' && c.label === 'Left') cost -= 0.15;
        if (side === 'L' && c.label === 'Right') cost -= 0.15;
        // Natural anatomical side relative to body midline
        if (side === 'R' && w.x < bodyCenterX + 0.05) cost -= 0.12;
        if (side === 'L' && w.x > bodyCenterX - 0.05) cost -= 0.12;
        return cost;
      }

      if (validCandidates.length === 1) {
        const c = validCandidates[0];
        const costR = assignmentCost(c, 'R');
        const costL = assignmentCost(c, 'L');
        if (costR <= costL) {
          results.rightHandLandmarks = c.lm;
          if (results.leftHandLandmarks && Math.hypot(results.leftHandLandmarks[0].x - c.lm[0].x, results.leftHandLandmarks[0].y - c.lm[0].y) < 0.12) {
            results.leftHandLandmarks = null;
          }
        } else {
          results.leftHandLandmarks = c.lm;
          if (results.rightHandLandmarks && Math.hypot(results.rightHandLandmarks[0].x - c.lm[0].x, results.rightHandLandmarks[0].y - c.lm[0].y) < 0.12) {
            results.rightHandLandmarks = null;
          }
        }
      } else if (validCandidates.length >= 2) {
        const c0 = validCandidates[0], c1 = validCandidates[1];
        const cost0R_1L = assignmentCost(c0, 'R') + assignmentCost(c1, 'L');
        const cost0L_1R = assignmentCost(c0, 'L') + assignmentCost(c1, 'R');
        if (cost0R_1L <= cost0L_1R) {
          results.rightHandLandmarks = c0.lm;
          results.leftHandLandmarks = c1.lm;
        } else {
          results.leftHandLandmarks = c0.lm;
          results.rightHandLandmarks = c1.lm;
        }
      }

      if (results.leftHandLandmarks && results.leftHandLandmarks[0]) {
        prevValidWrist.L = { x: results.leftHandLandmarks[0].x, y: results.leftHandLandmarks[0].y, ts: now };
      }
      if (results.rightHandLandmarks && results.rightHandLandmarks[0]) {
        prevValidWrist.R = { x: results.rightHandLandmarks[0].x, y: results.rightHandLandmarks[0].y, ts: now };
      }
    }

    // Stabilize Pose shoulders (p11, p12) when a raised hand occludes the shoulder/chest
    // so p12 never jumps onto the raised right wrist (yellow shoulder line & 76x3 normalization).
    function stabilizePoseShoulders(results) {
      if (!results.poseLandmarks || results.poseLandmarks.length < 17) return;
      const p0 = results.poseLandmarks[0];
      const p11 = results.poseLandmarks[11]; // Left shoulder (right side of raw un-mirrored image)
      const p12 = results.poseLandmarks[12]; // Right shoulder (left side of raw un-mirrored image)
      if (!p11 || !p12) return;

      const handPts = [];
      if (results.rightHandLandmarks && results.rightHandLandmarks.length >= 10) {
        handPts.push(results.rightHandLandmarks[0], results.rightHandLandmarks[9]);
      }
      if (results.leftHandLandmarks && results.leftHandLandmarks.length >= 10) {
        handPts.push(results.leftHandLandmarks[0], results.leftHandLandmarks[9]);
      }
      const p15 = results.poseLandmarks[15], p16 = results.poseLandmarks[16];
      if (p15 && p15.y < 0.85) handPts.push(p15);
      if (p16 && p16.y < 0.85) handPts.push(p16);

      const near11 = handPts.some(h => Math.hypot(h.x - p11.x, h.y - p11.y) < 0.17);
      const near12 = handPts.some(h => Math.hypot(h.x - p12.x, h.y - p12.y) < 0.17);
      const curWidth = Math.hypot(p11.x - p12.x, p11.y - p12.y);
      const tiltY = Math.abs(p11.y - p12.y);
      const widthBad = (curWidth < 0.15 || curWidth > 0.62) ||
        (stableShoulders.width > 0 && Math.abs(curWidth - stableShoulders.width) > 0.22 * stableShoulders.width);
      const jumped12 = stableShoulders.p12 && (Math.abs(p12.y - stableShoulders.p12.y) > 0.045 || Math.abs(p12.x - stableShoulders.p12.x) > 0.06);
      const jumped11 = stableShoulders.p11 && (Math.abs(p11.y - stableShoulders.p11.y) > 0.045 || Math.abs(p11.x - stableShoulders.p11.x) > 0.06);

      if (!near11 && !near12 && tiltY < 0.075 && !widthBad && p12.x < p11.x) {
        if (!stableShoulders.p11) {
          stableShoulders.p11 = { x: p11.x, y: p11.y, z: p11.z || 0.0 };
          stableShoulders.p12 = { x: p12.x, y: p12.y, z: p12.z || 0.0 };
          stableShoulders.width = curWidth;
        } else {
          const a = 0.20;
          stableShoulders.p11 = {
            x: (1 - a) * stableShoulders.p11.x + a * p11.x,
            y: (1 - a) * stableShoulders.p11.y + a * p11.y,
            z: (1 - a) * stableShoulders.p11.z + a * (p11.z || 0.0)
          };
          stableShoulders.p12 = {
            x: (1 - a) * stableShoulders.p12.x + a * p12.x,
            y: (1 - a) * stableShoulders.p12.y + a * p12.y,
            z: (1 - a) * stableShoulders.p12.z + a * (p12.z || 0.0)
          };
          stableShoulders.width = (1 - a) * stableShoulders.width + a * curWidth;
        }
        return;
      }

      // Lock occluded / jumped shoulder to stable reference (or symmetric counterpart across nose p0)
      if ((near12 || tiltY >= 0.075 || widthBad || jumped12) && !(near11 && !near12)) {
        if (stableShoulders.p12) {
          const dx = (!near11 && stableShoulders.p11) ? (p11.x - stableShoulders.p11.x) : 0.0;
          const dy = (!near11 && stableShoulders.p11) ? (p11.y - stableShoulders.p11.y) : 0.0;
          results.poseLandmarks[12] = {
            x: stableShoulders.p12.x + dx,
            y: stableShoulders.p12.y + dy,
            z: stableShoulders.p12.z,
            visibility: 0.95
          };
        } else if (p0) {
          results.poseLandmarks[12] = {
            x: Math.min(p12.x, 2.0 * p0.x - p11.x),
            y: p11.y,
            z: p11.z || 0.0,
            visibility: 0.95
          };
        }
      }
      if ((near11 || tiltY >= 0.075 || widthBad || jumped11) && near11) {
        if (stableShoulders.p11) {
          const r12 = results.poseLandmarks[12];
          const dx = (!near12 && stableShoulders.p12) ? (r12.x - stableShoulders.p12.x) : 0.0;
          const dy = (!near12 && stableShoulders.p12) ? (r12.y - stableShoulders.p12.y) : 0.0;
          results.poseLandmarks[11] = {
            x: stableShoulders.p11.x + dx,
            y: stableShoulders.p11.y + dy,
            z: stableShoulders.p11.z,
            visibility: 0.95
          };
        } else if (p0) {
          const r12 = results.poseLandmarks[12];
          results.poseLandmarks[11] = {
            x: Math.max(p11.x, 2.0 * p0.x - r12.x),
            y: r12.y,
            z: r12.z || 0.0,
            visibility: 0.95
          };
        }
      }
    }

    function extract76x3FromHolisticResults(results, mirrorFlip) {
      const pose33 = Array.from({ length: 33 }, () => [0.0, 0.0, 0.0]);
      const left21 = Array.from({ length: 21 }, () => [0.0, 0.0, 0.0]);
      const right21 = Array.from({ length: 21 }, () => [0.0, 0.0, 0.0]);
      let nose1 = [[0.0, 0.0, 0.0]];

      if (results.poseLandmarks && results.poseLandmarks.length >= 33) {
        for (let i = 0; i < 33; i++) {
          const lm = results.poseLandmarks[i];
          pose33[i] = [
            mirrorFlip ? (1.0 - lm.x) : lm.x,
            lm.y,
            lm.z || 0.0
          ];
        }
        if (mirrorFlip) {
          for (const [a, b] of POSE_LR_PAIRS) {
            const tmp = pose33[a];
            pose33[a] = pose33[b];
            pose33[b] = tmp;
          }
        }
        nose1 = [[pose33[0][0], pose33[0][1], pose33[0][2]]];
      }
      // keypoints_extractor.py uses face_landmarks[1] (nose tip of FaceMesh) for the 76th point
      if (results.faceLandmarks && results.faceLandmarks.length > 1) {
        const f = results.faceLandmarks[1];
        nose1 = [[mirrorFlip ? (1.0 - f.x) : f.x, f.y, f.z || 0.0]];
      }

      const rawLeft = mirrorFlip ? results.rightHandLandmarks : results.leftHandLandmarks;
      const rawRight = mirrorFlip ? results.leftHandLandmarks : results.rightHandLandmarks;

      if (rawLeft && rawLeft.length >= 21) {
        for (let i = 0; i < 21; i++) {
          const lm = rawLeft[i];
          left21[i] = [mirrorFlip ? (1.0 - lm.x) : lm.x, lm.y, lm.z || 0.0];
        }
      }
      if (rawRight && rawRight.length >= 21) {
        for (let i = 0; i < 21; i++) {
          const lm = rawRight[i];
          right21[i] = [mirrorFlip ? (1.0 - lm.x) : lm.x, lm.y, lm.z || 0.0];
        }
      }

      const hasLeft = rawLeft && rawLeft.length >= 21;
      const hasRight = rawRight && rawRight.length >= 21;

      // FAST-MOTION RECOVERY: when MediaPipe loses a hand for a few frames (motion blur / palm flip)
      // but the Pose wrist is still tracked, re-attach the last seen hand shape to the live Pose wrist.
      let poseVis = Array.from({ length: 33 }, () => 0.0);
      if (results.poseLandmarks && results.poseLandmarks.length >= 33) {
        poseVis = results.poseLandmarks.map(lm => (lm.visibility === undefined ? 1.0 : lm.visibility));
        if (mirrorFlip) {
          for (const [a, b] of POSE_LR_PAIRS) { const t = poseVis[a]; poseVis[a] = poseVis[b]; poseVis[b] = t; }
        }
      }
      const gL = recoverHandFromPose('L', left21, !!hasLeft, pose33, poseVis);
      const gR = recoverHandFromPose('R', right21, !!hasRight, pose33, poseVis);

      const frame76 = [...pose33, ...gL.hand, ...gR.hand, ...nose1];
      return {
        frame76,
        hasLeft: !!hasLeft || gL.ghost,
        hasRight: !!hasRight || gR.ghost,
        ghostLeft: gL.ghost ? gL.hand : null,
        ghostRight: gR.ghost ? gR.hand : null,
        rawLeft, rawRight
      };
    }

    // ---- Pose-guided hand recovery memory (per anatomical side) ----
    const GHOST_MAX_FRAMES = 4;   // Bridges short fast-motion blur dropouts without sticking stale hands
    const handMemory = { L: null, R: null };
    function recoverHandFromPose(side, hand21, has, pose33, poseVis) {
      const wi = side === 'L' ? 15 : 16;
      const pw = pose33[wi];
      const pwValid = (Math.abs(pw[0]) > 1e-6 || Math.abs(pw[1]) > 1e-6) && poseVis[wi] > 0.45;
      if (has) {
        const w = hand21[0];
        handMemory[side] = {
          rel: hand21.map(p => [p[0] - w[0], p[1] - w[1], p[2] - w[2]]),
          wrist: w.slice(),
          poseWrist: pwValid ? pw.slice() : null,
          lastVel: handMemory[side] && handMemory[side].wrist ? [w[0] - handMemory[side].wrist[0], w[1] - handMemory[side].wrist[1]] : [0, 0],
          age: 0
        };
        return { hand: hand21, ghost: false };
      }
      const m = handMemory[side];
      if (!m) return { hand: hand21, ghost: false };
      m.age += 1;
      // Hand really left the frame / was lowered -> stop recovering
      if (m.age > GHOST_MAX_FRAMES || (pwValid && pw[1] > 0.92)) {
        handMemory[side] = null;
        return { hand: hand21, ghost: false };
      }
      let nw;
      if (pwValid && m.poseWrist) {
        // Follow the Pose wrist trajectory (robust under motion blur)
        nw = [m.wrist[0] + (pw[0] - m.poseWrist[0]), m.wrist[1] + (pw[1] - m.poseWrist[1]), m.wrist[2]];
      } else if (m.age <= 2) {
        // No Pose wrist: short constant-velocity extrapolation
        nw = [m.wrist[0] + m.lastVel[0] * m.age, m.wrist[1] + m.lastVel[1] * m.age, m.wrist[2]];
      } else {
        return { hand: hand21, ghost: false };
      }
      return { hand: m.rel.map(r => [nw[0] + r[0], nw[1] + r[1], nw[2] + r[2]]), ghost: true };
    }

    function drawGhostHand(ctx, hand21, mirrorFlip, W, H) {
      if (!hand21) return;
      const pts = hand21.map(p => ({ x: mirrorFlip ? (1.0 - p[0]) : p[0], y: p[1] }));
      ctx.save();
      ctx.setLineDash([5, 4]);
      drawConnectors(ctx, pts, HAND_CONNECTIONS, { color: '#f97316', lineWidth: 2 });
      ctx.restore();
      ctx.fillStyle = '#f97316';
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText('⚡ bù tay nhanh', pts[0].x * W - 20, pts[0].y * H + 18);
    }

    async function resetActiveStream() {
      pendingSentenceFrames = [];
      pendingWordFrames = [];
      await fetch('/api/stream_reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: currentMode })
      });
      document.getElementById('liveResultContent').textContent = 'Đã xoá bộ đệm. Sẵn sàng nhận ký hiệu mới.';
    }

    async function captureCleanClipAndPredict() {
      if (!cameraObj || isRecordingClip) return;
      const btn = document.getElementById('btnRecordClip');
      btn.disabled = true;
      recordedClip76x3 = [];

      const durationSec = currentMode === 'word' ? 2.4 : 4.0;
      document.getElementById('camStatusText').textContent = '⏳ Chuẩn bị giơ tay... (1s)';
      await new Promise(r => setTimeout(r, 900));

      isRecordingClip = true;
      document.getElementById('camStatusText').textContent =
        `🔴 ĐANG QUAY (${durationSec}s)...`;
      await new Promise(r => setTimeout(r, durationSec * 1000));
      isRecordingClip = false;

      btn.disabled = false;
      if (recordedClip76x3.length < 5) {
        document.getElementById('camStatusText').textContent =
          '⚠️ Chưa thấy rõ bàn tay, hãy thử lại!';
        return;
      }

      lastCapturedClip76x3 = recordedClip76x3.slice();
      document.getElementById('camStatusText').textContent =
        `⏳ Đang phân tích ${lastCapturedClip76x3.length} khung hình...`;

      const sentence_engine = document.getElementById('sentEngineSelect') ? document.getElementById('sentEngineSelect').value : 'word_spotting';
      const res = await fetch('/api/webcam/predict_clip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: currentMode,
          sentence_engine,
          aspect: camAspect(),
          frames_76x3: lastCapturedClip76x3
        })
      });
      const data = await res.json();
      document.getElementById('liveTelemetry').textContent = JSON.stringify(data, null, 2);
      if (data.ok) {
        document.getElementById('camStatusText').textContent =
          `✅ Đã nhận diện (${data.captured_frames} frames)`;
        if (currentMode === 'word' && data.word_prediction) {
          document.getElementById('liveResultContent').innerHTML =
            renderWordBadge(data.word_prediction, `Clip Camera vừa quay (${data.captured_frames} frames)`);
        } else if (currentMode === 'sentence' && data.stream_update) {
          document.getElementById('liveResultContent').innerHTML =
            renderSentenceChips(data.stream_update.predicted_glosses, null, {
              vietnamese_sentence: data.stream_update.vietnamese_sentence,
              segments: data.stream_update.segments,
              engine: data.sentence_engine || sentence_engine
            });
        }
      } else {
        document.getElementById('camStatusText').textContent = `❌ Lỗi: ${data.error}`;
      }
    }

    async function _submitDirectTeachPayload(label, clipFrames) {
      const btn1 = document.getElementById('btnRecordAndTeach');
      const btn2 = document.getElementById('btnFinetuneCam');
      if (btn1) btn1.disabled = true;
      if (btn2) btn2.disabled = true;
      document.getElementById('webcamAdaptStatus').textContent =
        `⏳ Đang trích xuất đa cửa sổ chuyển động & dạy trực tiếp từ "${label}"...`;
      try {
        const res = await fetch('/api/webcam/finetune_clip', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            label,
            aspect: camAspect(),
            frames_76x3: (clipFrames && clipFrames.length >= 5) ? clipFrames : null
          })
        });
        const data = await res.json();
        document.getElementById('liveTelemetry').textContent = JSON.stringify(data, null, 2);
        if (data.ok) {
          if (data.webcam_sample_counts) {
            window.webcamSampleCounts = data.webcam_sample_counts;
            updateWebcamAdaptDropdown();
          }
          const sentMsg = data.sentence_model_updated ? ' + Mô hình Câu' : '';
          document.getElementById('webcamAdaptStatus').textContent =
            `✅ Đã dạy thành công từ "${label}" (Tổng: ${data.saved_count_for_label || 1} mẫu Camera, cập nhật Từ đơn${sentMsg} & Prototype)! Kiểm tra: ${data.verify_prediction.raw_top1_label} (${(data.verify_prediction.confidence * 100).toFixed(1)}%)`;
          document.getElementById('liveResultContent').innerHTML =
            renderWordBadge(data.verify_prediction, `Sau khi dạy trực tiếp mẫu Camera (${label})`);
        } else {
          document.getElementById('webcamAdaptStatus').textContent = `❌ Lỗi: ${data.error}`;
        }
      } finally {
        if (btn1) btn1.disabled = false;
        if (btn2) btn2.disabled = false;
      }
    }

    async function recordAndTeachDirectly() {
      const label = document.getElementById('webcamAdaptLabelSelect').value;
      if (!label) {
        alert('Hãy chọn nhãn từ cần dạy!');
        return;
      }
      if (!cameraObj) {
        alert('Hãy bấm "📷 1. Bật Camera" trước khi quay và dạy trực tiếp!');
        return;
      }
      if (isRecordingClip) return;
      const btn1 = document.getElementById('btnRecordAndTeach');
      const btnRec = document.getElementById('btnRecordClip');
      if (btn1) btn1.disabled = true;
      if (btnRec) btnRec.disabled = true;

      recordedClip76x3 = [];
      document.getElementById('webcamAdaptStatus').textContent =
        `⏳ Chuẩn bị giơ tay làm ký hiệu "${label}"... (0.8s)`;
      document.getElementById('camStatusText').textContent = `⏳ Chuẩn bị ký hiệu "${label}"...`;
      await new Promise(r => setTimeout(r, 800));

      isRecordingClip = true;
      document.getElementById('webcamAdaptStatus').textContent =
        `🔴 ĐANG GHI HÌNH MẪU "${label}" (2.4s) — Hãy thực hiện ký hiệu!`;
      document.getElementById('camStatusText').textContent = `🔴 ĐANG DẠY "${label}" (2.4s)...`;
      await new Promise(r => setTimeout(r, 2400));
      isRecordingClip = false;
      if (btnRec) btnRec.disabled = false;

      if (recordedClip76x3.length < 5) {
        if (btn1) btn1.disabled = false;
        document.getElementById('webcamAdaptStatus').textContent =
          '⚠️ Chưa thấy rõ bàn tay trong lúc quay, hãy thử lại!';
        document.getElementById('camStatusText').textContent = '🟢 Camera đang hoạt động';
        return;
      }
      lastCapturedClip76x3 = recordedClip76x3.slice();
      document.getElementById('camStatusText').textContent =
        `✅ Đã thu ${lastCapturedClip76x3.length} frames cho "${label}"`;
      await _submitDirectTeachPayload(label, lastCapturedClip76x3);
    }

    async function finetuneWithLastWebcamClip() {
      const label = document.getElementById('webcamAdaptLabelSelect').value;
      if (!label) {
        alert('Hãy chọn nhãn từ cần tinh chỉnh!');
        return;
      }
      // If user hasn't recorded a clip yet and camera is running, automatically record & teach!
      if (lastCapturedClip76x3.length < 5 && cameraObj && !isRecordingClip) {
        await recordAndTeachDirectly();
        return;
      }
      await _submitDirectTeachPayload(label, lastCapturedClip76x3.length >= 5 ? lastCapturedClip76x3 : null);
    }

    async function clearWebcamSamplesForLabel() {
      const label = document.getElementById('webcamAdaptLabelSelect').value;
      if (!label) return;
      const res = await fetch('/api/webcam/clear_samples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label })
      });
      const data = await res.json();
      if (data.ok) {
        window.webcamSampleCounts = data.webcam_sample_counts || {};
        updateWebcamAdaptDropdown();
        document.getElementById('webcamAdaptStatus').textContent =
          `🗑️ Đã xoá ${data.removed_files} mẫu Camera cũ của từ "${label}". Hãy bấm "🎯 Quay & Dạy trực tiếp (2.5s)" để dạy lại mẫu sạch!`;
      }
    }

    async function startWebcam() {
      const videoElement = document.getElementById('webcamVideo');
      const canvasElement = document.getElementById('webcamCanvas');
      const canvasCtx = canvasElement.getContext('2d');
      const elTelemetryDetails = document.getElementById('telemetryDetails');

      document.getElementById('camStatusText').textContent = '⏳ Đang tải MediaPipe Hands (BlazePalm) & Holistic...';
      if (typeof Hands !== 'undefined') {
        handsObj = new Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        handsObj.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.50,
          minTrackingConfidence: 0.50
        });
        handsObj.onResults((hRes) => {
          latestHandsResult = {
            multiHandLandmarks: (hRes && hRes.multiHandLandmarks) ? hRes.multiHandLandmarks : [],
            multiHandedness: (hRes && hRes.multiHandedness) ? hRes.multiHandedness : [],
            ts: performance.now()
          };
        });
      }

      holisticObj = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
      });
      holisticObj.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        refineFaceLandmarks: false,
        minDetectionConfidence: 0.55,
        minTrackingConfidence: 0.55
      });

      holisticObj.onResults((results) => {
        // 1) Replace Pose-cropped Holistic hand landmarks with BlazePalm (@mediapipe/hands) landmarks
        //    and discard any crumpled/folded-back hand skeletons.
        fuseBlazePalmHandsIntoHolistic(results);
        // 2) Lock Pose shoulders (p11, p12) when a raised hand occludes the shoulder/chest
        //    so p12 never jumps onto the right wrist.
        stabilizePoseShoulders(results);

        const mirrorFlip = document.getElementById('chkMirrorFlip').checked;
        const { frame76, hasLeft, hasRight, ghostLeft, ghostRight } = extract76x3FromHolisticResults(results, mirrorFlip);

        const pillL = document.getElementById('pillLeftHand');
        const pillR = document.getElementById('pillRightHand');
        if (pillL) pillL.className = hasLeft ? 'hand-pill active-left' : 'hand-pill';
        if (pillR) pillR.className = hasRight ? 'hand-pill active-right' : 'hand-pill';

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

        // Draw Pose upper-body shoulders (11 & 12)
        if (results.poseLandmarks && results.poseLandmarks.length > 12) {
          const p11 = results.poseLandmarks[11];
          const p12 = results.poseLandmarks[12];
          canvasCtx.strokeStyle = '#facc15';
          canvasCtx.lineWidth = 3;
          canvasCtx.beginPath();
          canvasCtx.moveTo(p11.x * canvasElement.width, p11.y * canvasElement.height);
          canvasCtx.lineTo(p12.x * canvasElement.width, p12.y * canvasElement.height);
          canvasCtx.stroke();
        }

        // Draw Left Hand (Green) and Right Hand (Cyan) according to current mirrorFlip setting
        const dispLeft = mirrorFlip ? results.rightHandLandmarks : results.leftHandLandmarks;
        const dispRight = mirrorFlip ? results.leftHandLandmarks : results.rightHandLandmarks;

        if (dispLeft) {
          drawConnectors(canvasCtx, dispLeft, HAND_CONNECTIONS, { color: '#22c55e', lineWidth: 2 });
          drawLandmarks(canvasCtx, dispLeft, { color: '#bbf7d0', lineWidth: 1, radius: 2 });
          const w = dispLeft[0];
          canvasCtx.fillStyle = '#22c55e';
          canvasCtx.font = 'bold 16px sans-serif';
          canvasCtx.fillText('L (Tay Trái)', w.x * canvasElement.width - 20, w.y * canvasElement.height - 10);
        }
        if (dispRight) {
          drawConnectors(canvasCtx, dispRight, HAND_CONNECTIONS, { color: '#38bdf8', lineWidth: 2 });
          drawLandmarks(canvasCtx, dispRight, { color: '#e0f2fe', lineWidth: 1, radius: 2 });
          const w = dispRight[0];
          canvasCtx.fillStyle = '#38bdf8';
          canvasCtx.font = 'bold 16px sans-serif';
          canvasCtx.fillText('R (Tay Phải)', w.x * canvasElement.width - 20, w.y * canvasElement.height - 10);
        }
        drawGhostHand(canvasCtx, ghostLeft, mirrorFlip, canvasElement.width, canvasElement.height);
        drawGhostHand(canvasCtx, ghostRight, mirrorFlip, canvasElement.width, canvasElement.height);
        canvasCtx.restore();

        if (isRecordingClip && (hasLeft || hasRight)) {
          recordedClip76x3.push(frame76);
        }

        // Queue EVERY frame so nothing is dropped while the server is decoding (fast signing needs every frame)
        if (!isRecordingClip) {
          if (currentMode === 'sentence') {
            pendingSentenceFrames.push(frame76);
            if (pendingSentenceFrames.length > 240) pendingSentenceFrames.splice(0, pendingSentenceFrames.length - 240);
          } else {
            pendingWordFrames.push(frame76);
            if (pendingWordFrames.length > 60) pendingWordFrames.splice(0, pendingWordFrames.length - 60);
          }
        }

        if (!isSendingFrame && !isRecordingClip) {
          isSendingFrame = true;
          const endpoint = currentMode === 'word' ? '/api/word/stream_frame' : '/api/sentence/stream_frame';
          const sentence_engine = document.getElementById('sentEngineSelect') ? document.getElementById('sentEngineSelect').value : 'word_spotting';
          const payload = {
            timestamp: performance.now(),
            aspect: camAspect(),
            sentence_engine
          };
          if (currentMode === 'sentence') {
            payload.frames_76x3 = pendingSentenceFrames.splice(0, pendingSentenceFrames.length);
          } else {
            payload.frames_76x3 = pendingWordFrames.splice(0, pendingWordFrames.length);
          }
          fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          })
            .then(resp => resp.json())
            .then(data => {
              if (data.ok) {
                const batchInfo = data.frames_received ? ` • Lô gửi: ${data.frames_received} frame` : '';
                document.getElementById('livePhaseText').textContent =
                  `${data.phase} • Phát hiện: ${data.hand_count} bàn tay • Bộ đệm: ${data.buffer_length} khung hình${batchInfo}`;
                if (currentMode === 'word' && data.word_prediction) {
                  document.getElementById('liveResultContent').innerHTML = renderWordBadge(data.word_prediction, 'Luồng Camera Trực tiếp');
                } else if (currentMode === 'sentence' && data.stream_update) {
                  document.getElementById('liveResultContent').innerHTML =
                    renderSentenceChips(data.stream_update.predicted_glosses, null, {
                      vietnamese_sentence: data.stream_update.vietnamese_sentence,
                      segments: data.stream_update.segments,
                      engine: sentence_engine
                    });
                }
                if (!elTelemetryDetails || elTelemetryDetails.open) {
                  document.getElementById('liveTelemetry').textContent = JSON.stringify(data, null, 2);
                }
              }
            })
            .catch(() => {})
            .finally(() => {
              isSendingFrame = false;
            });
        }
      });

      cameraObj = new Camera(videoElement, {
        onFrame: async () => {
          if (handsObj) {
            try { await handsObj.send({ image: videoElement }); } catch (e) {}
          }
          if (holisticObj) await holisticObj.send({ image: videoElement });
        },
        width: 640,
        height: 480
      });
      await cameraObj.start();
      try {
        const track = videoElement.srcObject && videoElement.srcObject.getVideoTracks
          ? videoElement.srcObject.getVideoTracks()[0] : null;
        if (track && track.applyConstraints) {
          try {
            await track.applyConstraints({ width: 640, height: 480, frameRate: { ideal: 60, min: 30 } });
          } catch (e1) {
            await track.applyConstraints({ frameRate: { ideal: 60 } });
          }
        }
      } catch (e) { /* keep default fps */ }
      document.getElementById('btnStartCam').disabled = true;
      document.getElementById('btnStopCam').disabled = false;
      document.getElementById('btnRecordClip').disabled = false;
      document.getElementById('camStatusText').textContent = '🟢 Camera đang hoạt động';
    }

    function stopWebcam() {
      if (cameraObj) {
        cameraObj.stop();
        cameraObj = null;
      }
      document.getElementById('btnStartCam').disabled = false;
      document.getElementById('btnStopCam').disabled = true;
      document.getElementById('btnRecordClip').disabled = true;
      document.getElementById('camStatusText').textContent = 'Camera đang tắt';
    }

    refreshStatus().then(() => scanNpyFolder());
  </script>
</body>
</html>
"""


def _enrich_word_prediction_with_top3(
    pred: Dict[str, Any], class_names: List[str]
) -> Dict[str, Any]:
    probs = pred.get("probabilities", [])
    indexed = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)[:3]
    pred["top3"] = [
        {
            "label": class_names[idx] if idx < len(class_names) else f"CLASS_{idx}",
            "prob": float(p),
        }
        for idx, p in indexed
    ]
    return pred


def _word_pred_from_probs(
    recognizer: VSLSingleSignRecognizer, probs: np.ndarray, latency_ms: float = 0.0
) -> Dict[str, Any]:
    """Build a prediction dict (same schema as `VSLSingleSignRecognizer.predict`) from a prob vector."""
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(probs)[::-1]
    top1 = int(order[0])
    p1 = float(probs[top1])
    p2 = float(probs[order[1]]) if len(probs) > 1 else 0.0
    raw_label = recognizer.class_names[top1]
    status, label = "ACCEPTED", raw_label
    if p1 < recognizer.unknown_threshold:
        status, label = "UNKNOWN", "UNKNOWN"
    elif len(probs) > 1 and (p1 - p2) < recognizer.ambiguity_margin:
        status, label = "AMBIGUOUS", "AMBIGUOUS"
    pred = {
        "predicted_label": label,
        "raw_top1_label": raw_label,
        "confidence": p1,
        "margin": p1 - p2,
        "status": status,
        "probabilities": [float(x) for x in probs.tolist()],
        "latency_ms": float(latency_ms),
    }
    return _enrich_word_prediction_with_top3(pred, recognizer.class_names)


def _clip_to_word_features(clip_np: np.ndarray) -> List[List[float]]:
    """Webcam `(T, 76, 3)` clip -> `(30, 138)` using EXACTLY the same function as `.npy` files."""
    return npy_sequence_to_word_138_features(clip_np, target_len=SINGLE_SIGN_SEQUENCE_LENGTH)


def _fuse_probs_with_webcam_prototypes(
    p_bi: np.ndarray,
    seq138: Any,
    class_names: List[str],
) -> np.ndarray:
    """
    Fuse BiLSTM probability vector `p_bi` with few-shot cosine similarity against the user's
    taught `STATE.webcam_prototypes`. When a live webcam sign matches a taught prototype (`sim >= 0.72`),
    the prototype boosts or corrects the prediction immediately.
    """
    if not STATE.webcam_prototypes:
        return p_bi
    desc = seq138_to_descriptor(seq138)
    n_cls = len(class_names)
    sims = np.full(n_cls, -1.0, dtype=np.float64)
    for ci, cls in enumerate(class_names):
        protos = STATE.webcam_prototypes.get(cls) or []
        if protos:
            s_vals = sorted((float(np.dot(desc, p)) for p in protos), reverse=True)[:2]
            sims[ci] = float(np.mean(s_vals))
    max_sim = float(np.max(sims))
    if max_sim < 0.72:
        return p_bi

    p_proto = np.zeros(n_cls, dtype=np.float64)
    for ci in range(n_cls):
        if sims[ci] >= max(0.65, max_sim - 0.10):
            p_proto[ci] = math.exp(32.0 * (sims[ci] - max_sim))
    s_sum = float(np.sum(p_proto))
    if s_sum <= 1e-9:
        return p_bi
    p_proto /= s_sum

    top_bi = int(np.argmax(p_bi))
    top_pr = int(np.argmax(p_proto))
    if top_bi == top_pr:
        # Both BiLSTM and Prototype agree -> boost confidence cleanly without diluting
        p_out = 0.65 * p_bi + 0.35 * p_proto
        p_out[top_bi] = max(float(p_bi[top_bi]), float(p_out[top_bi]), min(0.98, max_sim))
        return p_out / float(np.sum(p_out))

    # Prototype strongly matches a taught word -> blend to override old dataset camera-angle bias
    alpha = min(0.68, max(0.35, (max_sim - 0.70) * 2.6))
    p_fused = (1.0 - alpha) * p_bi + alpha * p_proto
    return p_fused / float(np.sum(p_fused))


def _predict_word_clip(clip_np: np.ndarray) -> Dict[str, Any]:
    assert STATE.word_recognizer is not None
    t0 = time.perf_counter()
    class_names = STATE.word_recognizer.class_names
    wins = extract_active_sign_windows_76x3(clip_np, fast_mode=True)[:2]
    best_probs: Optional[np.ndarray] = None
    best_conf = -1.0
    for w in wins:
        seq138 = _clip_to_word_features(w)
        raw = STATE.word_recognizer.predict(seq138)
        p_bi = np.asarray(raw["probabilities"], dtype=np.float64)
        p_fused = _fuse_probs_with_webcam_prototypes(p_bi, seq138, class_names)
        conf = float(np.max(p_fused))
        if conf > best_conf:
            best_conf = conf
            best_probs = p_fused
    assert best_probs is not None
    return _word_pred_from_probs(
        STATE.word_recognizer, best_probs, (time.perf_counter() - t0) * 1000.0
    )


class DualVSLRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html: str, status: int = 200) -> None:
        raw = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index.html"):
            self._send_html(INDEX_HTML)
            return

        if self.path == "/api/status":
            sent_words = STATE.get_sentence_classes()
            grammar_for_preview = (
                STATE.word_grammar
                if (STATE.word_grammar is not None and STATE.word_grammar.is_usable)
                else (
                    STATE.sentence_recognizer.grammar
                    if STATE.sentence_recognizer is not None
                    else None
                )
            )
            sample_sents = []
            if grammar_for_preview is not None and grammar_for_preview.is_usable:
                for g_seq in grammar_for_preview.sample_many(6, seed=19):
                    sample_sents.append(
                        {"glosses": g_seq, "vietnamese": glosses_to_vietnamese(g_seq)}
                    )
            self._send_json(
                {
                    "ok": True,
                    "default_npy_root": STATE.default_npy_root,
                    "word_model_loaded": STATE.word_recognizer is not None,
                    "word_classes": STATE.get_word_classes(),
                    "word_has_cm": os.path.exists(DEFAULT_WORD_CM_PNG),
                    "sentence_model_loaded": (
                        STATE.sentence_recognizer is not None
                        or STATE.sentence_segment_recognizer is not None
                    ),
                    "sentence_num_classes": len(sent_words) + 1 if sent_words else 0,
                    "sentence_classes": sent_words,
                    "sentence_has_cm": os.path.exists(DEFAULT_SENTENCE_CM_PNG),
                    "webcam_sample_counts": STATE.get_webcam_sample_counts(),
                    "grammar_templates": (
                        grammar_for_preview.realizable_templates()
                        if grammar_for_preview is not None
                        else []
                    ),
                    "grammar_examples": sample_sents,
                }
            )
            return

        if self.path.startswith("/api/word/confusion_matrix"):
            try:
                vis = compute_word_model_confusion_matrix()
                self._send_json({"ok": True, "data": vis})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=400)
            return

        if self.path.startswith("/api/sentence/confusion_matrix"):
            try:
                vis = compute_sentence_model_confusion_matrix()
                self._send_json({"ok": True, "data": vis})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=400)
            return

        self._send_json({"ok": False, "error": f"Unknown endpoint {self.path}"}, status=404)

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", "0"))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            body = {}

        try:
            if self.path == "/api/npy/scan":
                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                info = scan_npy_dataset_root(root_dir)
                STATE.default_npy_root = root_dir
                self._send_json({"ok": True, "info": info})
                return

            # =================================================================
            # PART 1: WORD RECOGNITION (BiLSTM 30x138) FROM .NPY — INDEPENDENT MODEL
            # =================================================================
            if self.path == "/api/word/train_npy":
                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                selected_folders = body.get("selected_folders")
                max_classes = int(body.get("max_classes", 8))
                max_files = int(body.get("max_files_per_class", 35))
                epochs = int(body.get("epochs", 35))
                batch_size = int(body.get("batch_size", 8))
                lr = float(body.get("lr", 1.5e-3))

                if selected_folders:
                    selected_folders = [normalize_single_gloss(f) for f in selected_folders]

                word_samples = load_word_samples_from_npy_root(
                    root_dir=root_dir,
                    selected_folders=selected_folders,
                    max_classes=max_classes,
                    max_files_per_class_split=max_files,
                )
                summary = train_single_sign_pipeline(
                    rows=word_samples,
                    output_model_path=DEFAULT_WORD_CKPT,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    seed=42,
                )
                STATE.reload_word_checkpoint(DEFAULT_WORD_CKPT, is_demo=False)
                self._send_json(
                    {
                        "ok": True,
                        "loaded_samples": len(word_samples),
                        "summary": summary,
                    }
                )
                return

            if self.path == "/api/word/predict_npy":
                STATE.ensure_word_model_loaded()
                assert STATE.word_recognizer is not None

                b64_str = str(body.get("npy_base64", ""))
                raw_bytes = base64.b64decode(b64_str)
                arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
                seq_138 = npy_sequence_to_word_138_features(arr)
                pred = STATE.word_recognizer.predict(seq_138)
                pred = _enrich_word_prediction_with_top3(
                    pred, STATE.word_recognizer.class_names
                )
                self._send_json(
                    {
                        "ok": True,
                        "filename": body.get("filename", "uploaded.npy"),
                        "trained_classes": STATE.word_recognizer.class_names,
                        "original_npy_shape": list(arr.shape),
                        "resampled_feature_shape": [len(seq_138), len(seq_138[0])],
                        "result": pred,
                    }
                )
                return

            if self.path == "/api/word/predict_random_test_npy":
                STATE.ensure_word_model_loaded()
                assert STATE.word_recognizer is not None

                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                test_dir = os.path.join(root_dir, "test")
                trained_classes = set(STATE.word_recognizer.class_names)

                # Pick a random .npy file from a folder matching one of the trained classes
                candidate_files = []
                if os.path.isdir(test_dir):
                    for folder in os.listdir(test_dir):
                        norm_f = normalize_single_gloss(folder)
                        if norm_f in trained_classes:
                            f_dir = os.path.join(test_dir, folder)
                            for fn in os.listdir(f_dir):
                                if fn.endswith(".npy"):
                                    candidate_files.append((norm_f, os.path.join(f_dir, fn)))

                if not candidate_files:
                    raise ValueError(
                        "Hãy bấm 'Huấn luyện BiLSTM Từ đơn từ thư mục .npy' trước để model học các lớp từ trong keypoints_splited."
                    )

                import random
                true_label, chosen_path = random.choice(candidate_files)
                arr = np.load(chosen_path, allow_pickle=False)
                seq_138 = npy_sequence_to_word_138_features(arr)
                pred = STATE.word_recognizer.predict(seq_138)
                pred = _enrich_word_prediction_with_top3(
                    pred, STATE.word_recognizer.class_names
                )
                self._send_json(
                    {
                        "ok": True,
                        "true_label": true_label,
                        "file_path": chosen_path,
                        "original_npy_shape": list(arr.shape),
                        "result": pred,
                    }
                )
                return

            if self.path == "/api/word/confusion_matrix":
                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                vis = compute_word_model_confusion_matrix(root_dir=root_dir)
                self._send_json({"ok": True, "data": vis})
                return

            if self.path == "/api/word/stream_frame":
                STATE.ensure_word_model_loaded()
                assert STATE.word_recognizer is not None

                batch = parse_webcam_frames_batch_76x3(body)
                for f_76, h_cnt in batch:
                    STATE.word_step_counter += 1
                    if h_cnt > 0:
                        # If user had lowered hands for >= 8 frames and raises hands again, start fresh sign window
                        if STATE.word_missing_hand_streak >= 8 and len(STATE.word_live_buffer_76x3) >= 10:
                            STATE.word_live_buffer_76x3 = []
                            STATE.word_ema_probs = None
                            STATE.word_peak_prediction = None
                        STATE.word_missing_hand_streak = 0
                        STATE.word_live_buffer_76x3.append(f_76)
                        if len(STATE.word_live_buffer_76x3) > 54:
                            STATE.word_live_buffer_76x3 = STATE.word_live_buffer_76x3[-54:]
                    else:
                        STATE.word_missing_hand_streak += 1
                        # Keep short mid-sign dropout frames (<= 5 frames) in buffer so interpolation can bridge them
                        if 0 < len(STATE.word_live_buffer_76x3) and STATE.word_missing_hand_streak <= 5:
                            STATE.word_live_buffer_76x3.append(f_76)
                            if len(STATE.word_live_buffer_76x3) > 54:
                                STATE.word_live_buffer_76x3 = STATE.word_live_buffer_76x3[-54:]

                frame_76x3, hand_count = batch[-1]
                buf_len = len(STATE.word_live_buffer_76x3)
                if hand_count > 0 and buf_len >= 5:
                    phase = "SIGNING"
                elif hand_count > 0:
                    phase = "SIGN_START"
                elif STATE.word_missing_hand_streak <= 3 and buf_len >= 5:
                    phase = "SIGN_END"
                else:
                    phase = "REST"

                should_infer = (
                    (hand_count > 0 and buf_len >= 5)
                    or (hand_count == 0 and STATE.word_missing_hand_streak == 2 and buf_len >= 5)
                )

                if should_infer:
                    full_buf = np.stack(STATE.word_live_buffer_76x3[-48:], axis=0)
                    STATE.last_captured_clip_76x3 = full_buf
                    cur_pred = _predict_word_clip(full_buf)
                    if buf_len >= 22:
                        rec_pred = _predict_word_clip(full_buf[-28:])
                        if rec_pred["confidence"] > cur_pred["confidence"] + 0.04:
                            cur_pred = rec_pred
                    p_vec = np.asarray(cur_pred["probabilities"], dtype=np.float64)
                    if STATE.word_ema_probs is None or len(STATE.word_ema_probs) != len(p_vec):
                        STATE.word_ema_probs = p_vec
                    else:
                        STATE.word_ema_probs = 0.65 * p_vec + 0.35 * STATE.word_ema_probs
                    cand_pred = _word_pred_from_probs(
                        STATE.word_recognizer, STATE.word_ema_probs, cur_pred.get("latency_ms", 0.0)
                    )

                    # Detect if hand is currently dropping near the bottom edge at the end of a sign
                    active_ys = []
                    if np.any(np.abs(frame_76x3[33:54]) > 1e-6):
                        active_ys.append(float(frame_76x3[33, 1]))
                    if np.any(np.abs(frame_76x3[54:75]) > 1e-6):
                        active_ys.append(float(frame_76x3[54, 1]))
                    is_lowering = (hand_count == 0) or (bool(active_ys) and min(active_ys) > 0.87)

                    if cand_pred["status"] == "ACCEPTED" and cand_pred["confidence"] >= 0.78 and not is_lowering:
                        if (
                            STATE.word_peak_prediction is None
                            or cand_pred["confidence"] >= STATE.word_peak_prediction["confidence"] - 0.05
                        ):
                            STATE.word_peak_prediction = cand_pred

                    if (
                        is_lowering
                        and STATE.word_peak_prediction is not None
                        and cand_pred["confidence"] < STATE.word_peak_prediction["confidence"] - 0.08
                    ):
                        STATE.last_word_prediction = STATE.word_peak_prediction
                    else:
                        STATE.last_word_prediction = cand_pred

                self._send_json(
                    {
                        "ok": True,
                        "phase": phase,
                        "hand_count": hand_count,
                        "frames_received": len(batch),
                        "buffer_length": len(STATE.word_live_buffer_76x3),
                        "word_prediction": STATE.last_word_prediction,
                    }
                )
                return

            # =================================================================
            # PART 2: CONTINUOUS SENTENCE RECOGNITION — INDEPENDENT MODELS (CAN RE-TRAIN ON ANY DATA)
            # =================================================================
            if self.path == "/api/sentence/train_npy":
                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                selected_folders = body.get("selected_folders")
                max_classes = int(body.get("max_classes", 12))
                sentences_per_split = int(body.get("sentences_per_split", 120))
                epochs = int(body.get("epochs", 30))
                batch_size = int(body.get("batch_size", 8))
                use_grammar = bool(body.get("use_grammar", True))

                if selected_folders:
                    selected_folders = [normalize_single_gloss(f) for f in selected_folders]
                    max_classes = max(max_classes, len(selected_folders))
                else:
                    word_cls = STATE.get_word_classes()
                    if word_cls:
                        selected_folders = word_cls
                        max_classes = len(word_cls)

                # 1. Train Part 2's dedicated Abrupt-Transition Segment Encoder (vsl_sentence_segment.pth)
                seg_samples = load_word_samples_from_npy_root(
                    root_dir=root_dir,
                    selected_folders=selected_folders,
                    max_classes=max_classes,
                    max_files_per_class_split=90,
                )
                seg_summary = train_single_sign_pipeline(
                    rows=seg_samples,
                    output_model_path=DEFAULT_SENTENCE_SEGMENT_CKPT,
                    epochs=max(35, min(45, epochs)),
                    batch_size=batch_size,
                    lr=1.5e-3,
                    seed=42,
                )

                # 2. Train Part 2's 2-Layer BiLSTM CTC model (vsl_sentence_ctc.pth) on continuous sentences
                samples, custom_vocab = build_continuous_sentence_samples_from_npy_root(
                    root_dir=root_dir,
                    selected_folders=selected_folders,
                    max_classes=max_classes,
                    sentences_per_split=sentences_per_split,
                    transition_frames=1,
                    max_transition_frames=3,
                    use_grammar=use_grammar,
                )
                summary = train_ctc_pipeline(
                    raw_samples=samples,
                    output_model_path=DEFAULT_SENTENCE_CKPT,
                    vocab=custom_vocab,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=1.5e-3,
                    seed=42,
                )
                export_info = export_model_to_torchscript(
                    DEFAULT_SENTENCE_CKPT, DEFAULT_SENTENCE_PT
                )
                STATE.reload_sentence_checkpoint(
                    DEFAULT_SENTENCE_CKPT, DEFAULT_SENTENCE_SEGMENT_CKPT, is_demo=False
                )
                vocab_words = STATE.get_sentence_classes()
                grammar = SentenceGrammar(vocab_words)
                self._send_json(
                    {
                        "ok": True,
                        "built_sentences": len(samples),
                        "use_grammar": use_grammar,
                        "grammar_templates": grammar.realizable_templates(),
                        "vocabulary": vocab_words,
                        "segment_encoder_accuracy": seg_summary.get("test_metrics", {}).get("accuracy"),
                        "summary": summary,
                        "torchscript_export": export_info,
                    }
                )
                return

            if self.path == "/api/sentence/predict_npy":
                STATE.ensure_sentence_model_loaded()
                decode_method = str(body.get("decode_method", "word_spotting"))
                files_list = body.get("files") or []
                if not files_list:
                    raise ValueError("Chưa có file .npy nào được gửi lên.")

                raw_clips: List[np.ndarray] = []
                combined_212: List[List[float]] = []
                file_shapes = []
                for idx, f_obj in enumerate(files_list):
                    raw_bytes = base64.b64decode(str(f_obj.get("npy_base64", "")))
                    arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
                    raw_clips.append(arr)
                    file_shapes.append({"filename": f_obj.get("filename"), "shape": list(arr.shape)})
                    seq212 = preprocess_npy_clip_for_ctc(arr, target_clip_frames=16)
                    if idx > 0 and len(combined_212) > 0 and len(seq212) > 0:
                        prev_v = combined_212[-1]
                        next_v = seq212[0]
                        for step in range(1, 3):
                            a = step / 3.0
                            combined_212.append([prev_v[d] + a * (next_v[d] - prev_v[d]) for d in range(212)])
                    combined_212.extend(seq212)

                if decode_method == "word_spotting":
                    if STATE.word_spotting_decoder is None:
                        STATE._rebuild_sentence_spotting()
                    assert STATE.word_spotting_decoder is not None
                    full_clip_76x3 = concat_raw_clips_76x3(raw_clips, transition_frames=1)
                    ws_res = STATE.word_spotting_decoder.decode(full_clip_76x3)
                    ws_res["vietnamese_sentence"] = glosses_to_vietnamese(ws_res["predicted_glosses"])
                    ws_res["decode_method"] = "word_spotting"
                    self._send_json(
                        {
                            "ok": True,
                            "uploaded_files": file_shapes,
                            "trained_vocabulary": STATE.get_sentence_classes(),
                            "total_sentence_frames": int(full_clip_76x3.shape[0]),
                            "result": ws_res,
                        }
                    )
                    return

                assert STATE.sentence_recognizer is not None
                res = STATE.sentence_recognizer.predict_from_features(
                    combined_212, decode_method=decode_method, beam_width=8
                )
                printable = {
                    k: v
                    for k, v in res.items()
                    if k not in ("frame_probabilities", "frame_log_probabilities")
                }
                self._send_json(
                    {
                        "ok": True,
                        "uploaded_files": file_shapes,
                        "trained_vocabulary": STATE.get_sentence_classes(),
                        "total_sentence_frames": len(combined_212),
                        "result": printable,
                    }
                )
                return

            if self.path == "/api/sentence/predict_random_test_npy":
                STATE.ensure_sentence_model_loaded()
                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                decode_method = str(body.get("decode_method", "word_spotting"))

                if decode_method == "word_spotting":
                    if STATE.word_spotting_decoder is None:
                        STATE._rebuild_sentence_spotting()
                    assert STATE.word_spotting_decoder is not None
                    sent_classes = STATE.get_sentence_classes()
                    raw_sents = build_raw_sentence_clips_from_npy_root(
                        root_dir=root_dir,
                        class_names=sent_classes,
                        num_sentences=1,
                        split="test",
                        seed=int(np.random.randint(1, 100000)),
                        use_grammar=True,
                        transition_frames=1,
                    )
                    chosen_raw = raw_sents[0]
                    ws_res = STATE.word_spotting_decoder.decode(chosen_raw["clip"])
                    ws_res["vietnamese_sentence"] = glosses_to_vietnamese(ws_res["predicted_glosses"])
                    self._send_json(
                        {
                            "ok": True,
                            "decode_method": "word_spotting",
                            "trained_vocabulary": sent_classes,
                            "true_glosses": chosen_raw["glosses"],
                            "true_vietnamese": glosses_to_vietnamese(chosen_raw["glosses"]),
                            "source_files": chosen_raw["files"],
                            "total_frames": int(chosen_raw["clip"].shape[0]),
                            "result": ws_res,
                            "greedy_result": {
                                "predicted_glosses": ws_res["predicted_glosses"],
                                "vietnamese_sentence": ws_res["vietnamese_sentence"],
                                "segments": ws_res.get("segments", []),
                                "latency_ms": ws_res["latency_ms"],
                            },
                        }
                    )
                    return

                assert STATE.sentence_recognizer is not None
                vocab_words = STATE.get_sentence_classes()
                samples, _ = build_continuous_sentence_samples_from_npy_root(
                    root_dir=root_dir,
                    selected_folders=vocab_words,
                    sentences_per_split=6,
                    seed=int(np.random.randint(1, 100000)),
                    use_grammar=True,
                )
                test_candidates = [s for s in samples if s["speakerId"] == "spk_npy_test"] or samples
                chosen = test_candidates[0]
                feat_seq = [f["combinedFeatureVector"] for f in chosen["frames"]]

                greedy_res = STATE.sentence_recognizer.predict_from_features(
                    feat_seq, decode_method="greedy"
                )
                beam_res = STATE.sentence_recognizer.predict_from_features(
                    feat_seq, decode_method="beam_search", beam_width=5
                )
                lm_res = STATE.sentence_recognizer.predict_from_features(
                    feat_seq, decode_method="beam_search_lm", beam_width=8
                )
                primary_res = lm_res if decode_method == "beam_search_lm" else (
                    beam_res if decode_method == "beam_search" else greedy_res
                )

                self._send_json(
                    {
                        "ok": True,
                        "decode_method": decode_method,
                        "trained_vocabulary": vocab_words,
                        "true_glosses": chosen["glossSequence"],
                        "true_vietnamese": glosses_to_vietnamese(chosen["glossSequence"]),
                        "total_frames": len(feat_seq),
                        "result": {
                            "predicted_glosses": primary_res["predicted_glosses"],
                            "vietnamese_sentence": primary_res["vietnamese_sentence"],
                            "n_best": primary_res.get("n_best"),
                            "latency_ms": primary_res["latency_ms"],
                        },
                        "greedy_result": {
                            "predicted_glosses": primary_res["predicted_glosses"],
                            "vietnamese_sentence": primary_res["vietnamese_sentence"],
                            "latency_ms": primary_res["latency_ms"],
                        },
                        "beam_search_result": {
                            "predicted_glosses": beam_res["predicted_glosses"],
                            "vietnamese_sentence": beam_res["vietnamese_sentence"],
                            "latency_ms": beam_res["latency_ms"],
                        },
                        "beam_search_lm_result": {
                            "predicted_glosses": lm_res["predicted_glosses"],
                            "vietnamese_sentence": lm_res["vietnamese_sentence"],
                            "n_best": lm_res.get("n_best"),
                            "latency_ms": lm_res["latency_ms"],
                        },
                    }
                )
                return

            if self.path == "/api/sentence/confusion_matrix":
                root_dir = str(body.get("root_dir") or STATE.default_npy_root)
                vis = compute_sentence_model_confusion_matrix(root_dir=root_dir)
                self._send_json({"ok": True, "data": vis})
                return

            if self.path == "/api/sentence/stream_frame":
                STATE.ensure_sentence_model_loaded()
                batch = parse_webcam_frames_batch_76x3(body)
                hand_count = int(batch[-1][1])
                sentence_engine = str(body.get("sentence_engine", "word_spotting"))
                STATE.sentence_step_counter += len(batch)

                if sentence_engine == "word_spotting":
                    if STATE.sentence_abrupt_tracker is None:
                        STATE._rebuild_sentence_spotting()
                    assert STATE.sentence_abrupt_tracker is not None
                    trk = STATE.sentence_abrupt_tracker
                    track_res = trk.push_frames([f for f, _ in batch], [h for _, h in batch])
                    if len(trk.active_frames) >= 4:
                        STATE.last_captured_clip_76x3 = np.stack(trk.active_frames, axis=0)
                    glosses = track_res.get("predicted_glosses", [])
                    if glosses or STATE.last_sentence_prediction is None:
                        STATE.last_sentence_prediction = {
                            "predicted_glosses": glosses,
                            "predicted_sentence": " ".join(glosses),
                            "vietnamese_sentence": glosses_to_vietnamese(glosses),
                            "segments": track_res.get("segments", []),
                            "confidence": float(track_res.get("confidence", 0.0)),
                            "latency_ms": float(track_res.get("decode_ms", 0.0)),
                            "sentence_completed": bool(track_res.get("sentence_completed")),
                            "engine": "word_spotting",
                        }
                    if track_res.get("sentence_completed"):
                        phase = "SENTENCE_DONE (Đã chốt câu)"
                    elif hand_count > 0:
                        phase = "SIGNING (Ghi trọn câu, không cắt cửa sổ)"
                    else:
                        phase = "REST"
                    self._send_json(
                        {
                            "ok": True,
                            "phase": phase,
                            "hand_count": hand_count,
                            "frames_received": len(batch),
                            "buffer_length": int(track_res.get("active_window_frames", 0)),
                            "stream_update": STATE.last_sentence_prediction,
                        }
                    )
                    return

                # ---- CTC engine: utterance-level buffer (whole sentence, no sliding window) ----
                ended = False
                new_frames = 0
                for frame_76x3, h_cnt in batch:
                    if h_cnt > 0:
                        if STATE.sentence_missing_hand_streak >= 3 and STATE.sentence_ctc_done:
                            STATE.sentence_live_buffer_76x3 = []
                            STATE.sentence_ctc_done = False
                        STATE.sentence_missing_hand_streak = 0
                        STATE.sentence_live_buffer_76x3.append(frame_76x3)
                        new_frames += 1
                    else:
                        STATE.sentence_missing_hand_streak += 1
                        if STATE.sentence_live_buffer_76x3 and STATE.sentence_missing_hand_streak <= 2:
                            STATE.sentence_live_buffer_76x3.append(frame_76x3)
                        if (
                            STATE.sentence_missing_hand_streak == 3
                            and len(STATE.sentence_live_buffer_76x3) >= 8
                            and not STATE.sentence_ctc_done
                        ):
                            ended = True
                    if len(STATE.sentence_live_buffer_76x3) > 360:
                        STATE.sentence_live_buffer_76x3 = STATE.sentence_live_buffer_76x3[-360:]
                STATE.sentence_ctc_new_frames += new_frames

                buf_len = len(STATE.sentence_live_buffer_76x3)
                phase = "SIGNING (Ghi trọn câu)" if hand_count > 0 else ("SENTENCE_DONE" if STATE.sentence_ctc_done or ended else "REST")

                if buf_len >= 8 and (ended or (hand_count > 0 and STATE.sentence_ctc_new_frames >= 6)):
                    STATE.sentence_ctc_new_frames = 0
                    clip_np = np.stack(STATE.sentence_live_buffer_76x3, axis=0)
                    STATE.last_captured_clip_76x3 = clip_np
                    assert STATE.sentence_recognizer is not None
                    res = ctc_multiscale_decode(STATE.sentence_recognizer, clip_np, beam_width=8 if ended else 5)
                    glosses = res.get("predicted_glosses", [])
                    STATE.last_sentence_prediction = {
                        "predicted_glosses": glosses,
                        "predicted_sentence": " ".join(glosses),
                        "vietnamese_sentence": glosses_to_vietnamese(glosses),
                        "confidence": float(res.get("confidence", 0.0)),
                        "latency_ms": float(res.get("latency_ms", 0.0)),
                        "best_target_len": res.get("best_target_len"),
                        "sentence_completed": bool(ended),
                        "engine": "beam_search_lm",
                    }
                    if ended:
                        STATE.sentence_ctc_done = True

                self._send_json(
                    {
                        "ok": True,
                        "phase": phase,
                        "hand_count": hand_count,
                        "frames_received": len(batch),
                        "buffer_length": buf_len,
                        "stream_update": STATE.last_sentence_prediction,
                    }
                )
                return

            if self.path == "/api/webcam/predict_clip":
                mode = str(body.get("mode", "word"))
                sentence_engine = str(body.get("sentence_engine", "word_spotting"))
                clip_np = parse_webcam_clip_76x3(body)
                if clip_np is None:
                    raise ValueError("Cần ít nhất 4 frame có bàn tay để nhận diện.")
                STATE.last_captured_clip_76x3 = clip_np

                if mode == "word":
                    STATE.ensure_word_model_loaded()
                    assert STATE.word_recognizer is not None
                    pred = _predict_word_clip(clip_np)
                    STATE.last_word_prediction = pred
                    self._send_json(
                        {
                            "ok": True,
                            "mode": "word",
                            "captured_frames": int(clip_np.shape[0]),
                            "word_prediction": pred,
                        }
                    )
                    return
                else:
                    STATE.ensure_sentence_model_loaded()
                    if sentence_engine == "word_spotting":
                        if STATE.word_spotting_decoder is None:
                            STATE._rebuild_sentence_spotting()
                        assert STATE.word_spotting_decoder is not None
                        ws_res = STATE.word_spotting_decoder.decode(clip_np)
                        glosses = ws_res.get("predicted_glosses", [])
                        stream_update = {
                            "predicted_glosses": glosses,
                            "predicted_sentence": " ".join(glosses),
                            "vietnamese_sentence": glosses_to_vietnamese(glosses),
                            "segments": ws_res.get("segments", []),
                            "confidence": float(ws_res.get("confidence", 0.0)),
                            "latency_ms": float(ws_res.get("latency_ms", 0.0)),
                            "engine": "word_spotting",
                        }
                    else:
                        assert STATE.sentence_recognizer is not None
                        res = ctc_multiscale_decode(STATE.sentence_recognizer, clip_np, beam_width=8)
                        glosses = res.get("predicted_glosses", [])
                        stream_update = {
                            "predicted_glosses": glosses,
                            "predicted_sentence": " ".join(glosses),
                            "vietnamese_sentence": glosses_to_vietnamese(glosses),
                            "confidence": float(res.get("confidence", 0.0)),
                            "latency_ms": float(res.get("latency_ms", 0.0)),
                            "engine": "beam_search_lm",
                        }
                    STATE.last_sentence_prediction = stream_update
                    self._send_json(
                        {
                            "ok": True,
                            "mode": "sentence",
                            "captured_frames": int(clip_np.shape[0]),
                            "stream_update": stream_update,
                        }
                    )
                    return

            if self.path == "/api/webcam/clear_samples":
                label = normalize_single_gloss(str(body.get("label", "")))
                if not label:
                    raise ValueError("Vui lòng chọn nhãn từ cần xoá mẫu.")
                label_dir = os.path.join(WEBCAM_NPY_DIR, label)
                removed = 0
                if os.path.isdir(label_dir):
                    for fn in os.listdir(label_dir):
                        if fn.endswith(".npy"):
                            fpath = os.path.join(label_dir, fn)
                            try:
                                os.remove(fpath)
                                STATE.webcam_file_cache.pop(fpath, None)
                                removed += 1
                            except Exception:
                                pass
                STATE.webcam_prototypes.pop(label, None)
                self._send_json(
                    {
                        "ok": True,
                        "label": label,
                        "removed_files": removed,
                        "webcam_sample_counts": STATE.get_webcam_sample_counts(),
                    }
                )
                return

            if self.path == "/api/webcam/finetune_clip":
                STATE.ensure_word_model_loaded()
                assert STATE.word_recognizer is not None

                label = normalize_single_gloss(str(body.get("label", "")))
                if not label:
                    raise ValueError("Vui lòng chọn nhãn từ để tinh chỉnh.")
                if label not in STATE.word_recognizer.class_names:
                    raise ValueError(
                        f"Nhãn '{label}' không nằm trong danh sách lớp hiện tại: {STATE.word_recognizer.class_names}"
                    )

                clip_np = parse_webcam_clip_76x3(body)
                if clip_np is not None:
                    pass
                elif STATE.last_captured_clip_76x3 is not None and len(STATE.last_captured_clip_76x3) >= 4:
                    clip_np = STATE.last_captured_clip_76x3
                elif len(STATE.word_live_buffer_76x3) >= 4:
                    clip_np = np.stack(STATE.word_live_buffer_76x3[-45:], axis=0)
                else:
                    raise ValueError("Chưa có dữ liệu Webcam vừa quay để lưu & tinh chỉnh.")

                clip_np = np.nan_to_num(clip_np, nan=0.0, posinf=0.0, neginf=0.0)
                # If clip_np came from a long sentence buffer (>58 frames), trim to its active sign core before saving
                active_wins_new = extract_active_sign_windows_76x3(clip_np, fast_mode=False)
                clip_to_save = active_wins_new[0] if len(clip_np) > 58 else clip_np

                label_dir = os.path.join(WEBCAM_NPY_DIR, label)
                os.makedirs(label_dir, exist_ok=True)
                save_path = os.path.join(label_dir, f"webcam_{int(time.time() * 1000)}.npy")
                np.save(save_path, clip_to_save)

                import random as _random

                class_names = STATE.word_recognizer.class_names
                rng = _random.Random(int(time.time()))
                xs: List[List[List[float]]] = []
                ys: List[int] = []
                taught_classes = set()
                new_proto_bank: Dict[str, List[np.ndarray]] = {}

                # (a) Multi-window + augmented webcam clips across all taught classes (cached for speed)
                seen_hashes = set()
                for cls in class_names:
                    cls_dir = os.path.join(WEBCAM_NPY_DIR, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    files = sorted(
                        os.path.join(cls_dir, fn)
                        for fn in os.listdir(cls_dir)
                        if fn.endswith(".npy")
                    )[-6:]
                    if not files:
                        continue
                    taught_classes.add(cls)
                    c_idx = class_names.index(cls)
                    cls_descs: List[np.ndarray] = []

                    for fi, fpath in enumerate(files):
                        is_latest = (fpath == save_path) or (fi == len(files) - 1 and cls == label)
                        cached = STATE.webcam_file_cache.get(fpath)
                        if cached is not None and not is_latest and cached.get("aug_seqs"):
                            for s138 in cached["seqs"]:
                                xs.append(s138)
                                ys.append(c_idx)
                            for s138 in cached["aug_seqs"]:
                                xs.append(s138)
                                ys.append(c_idx)
                            cls_descs.extend(cached.get("descs", []))
                            continue

                        try:
                            w_arr = np.load(fpath, allow_pickle=False)
                        except Exception:
                            continue
                        if len(w_arr) > 85 and len(files) > 1 and fpath != save_path:
                            continue
                        h = hash(w_arr.tobytes())
                        if h in seen_hashes and fpath != save_path:
                            continue
                        seen_hashes.add(h)

                        wins = (
                            active_wins_new
                            if fpath == save_path
                            else extract_active_sign_windows_76x3(w_arr, fast_mode=not is_latest)
                        )
                        f_seqs: List[List[List[float]]] = []
                        f_augs: List[List[List[float]]] = []
                        f_descs: List[np.ndarray] = []
                        for idx_w, win in enumerate(wins):
                            seq138 = _clip_to_word_features(win)
                            f_seqs.append(seq138)
                            xs.append(seq138)
                            ys.append(c_idx)
                            if idx_w < 5:
                                d = seq138_to_descriptor(seq138)
                                f_descs.append(d)
                                cls_descs.append(d)
                            n_aug = 2 if (is_latest and idx_w < 3) else (1 if idx_w == 0 else 0)
                            for _ in range(n_aug):
                                aug138 = _clip_to_word_features(augment_npy_76x3(win, rng))
                                f_augs.append(aug138)
                                xs.append(aug138)
                                ys.append(c_idx)
                        STATE.webcam_file_cache[fpath] = {
                            "label": cls,
                            "seqs": f_seqs,
                            "aug_seqs": f_augs,
                            "descs": f_descs,
                        }
                    if cls_descs:
                        new_proto_bank[cls] = cls_descs

                STATE.webcam_prototypes = new_proto_bank

                # (b) Balanced replay from dataset: keep fewer old dataset samples (5) for taught classes
                #     so the old dataset camera angle never overrides the user's taught webcam sign!
                if STATE.word_replay_cache is None:
                    STATE.word_replay_cache = []
                    if os.path.isdir(STATE.default_npy_root):
                        try:
                            replay = load_word_samples_from_npy_root(
                                root_dir=STATE.default_npy_root,
                                selected_folders=class_names,
                                max_classes=len(class_names),
                                max_files_per_class_split=22,
                                min_train_samples_per_class=0,
                                seed=99,
                            )
                            STATE.word_replay_cache = [
                                (s["sequence"], s["label"])
                                for s in replay
                                if s.get("split") == "train" and s["label"] in class_names
                            ]
                        except Exception as e:
                            print(f"[FINETUNE] replay load failed: {e}")

                per_cls_replay = {cls: 0 for cls in class_names}
                for seq, r_label in STATE.word_replay_cache:
                    cls = r_label if isinstance(r_label, str) else class_names[int(r_label)]
                    if cls not in class_names:
                        continue
                    limit = 5 if cls in taught_classes else 18
                    if per_cls_replay[cls] < limit:
                        xs.append(seq)
                        ys.append(class_names.index(cls))
                        per_cls_replay[cls] += 1

                def _run_fast_bilstm_adaptation(rec: VSLSingleSignRecognizer, ckpt_path: str) -> bool:
                    rec_classes = list(rec.class_names)
                    xs_rec, ys_rec = [], []
                    for x_item, y_idx in zip(xs, ys):
                        cls_name = class_names[int(y_idx)]
                        if cls_name in rec_classes:
                            xs_rec.append(x_item)
                            ys_rec.append(rec_classes.index(cls_name))
                    if len(xs_rec) < 4:
                        return False
                    dev = rec.device
                    mdl = rec.model
                    x_t = torch.tensor(np.asarray(xs_rec, dtype=np.float32), device=dev)
                    y_t = torch.tensor(ys_rec, dtype=torch.long, device=dev)
                    opt = torch.optim.AdamW(mdl.parameters(), lr=8e-4, weight_decay=1e-4)
                    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.03)
                    mdl.train()
                    n_s = x_t.shape[0]
                    for _ in range(14):
                        perm = torch.randperm(n_s, device=dev)
                        for b in range(0, n_s, 24):
                            idx = perm[b : b + 24]
                            if idx.numel() < 2:
                                continue
                            opt.zero_grad()
                            loss = crit(mdl.forward_logits(x_t[idx]), y_t[idx])
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 2.0)
                            opt.step()
                    mdl.eval()
                    if os.path.exists(ckpt_path):
                        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                        ck["model_state_dict"] = {k: v.detach().cpu() for k, v in mdl.state_dict().items()}
                        torch.save(ck, ckpt_path)
                    return True

                _run_fast_bilstm_adaptation(STATE.word_recognizer, DEFAULT_WORD_CKPT)

                sentence_model_updated = False
                if (
                    STATE.sentence_segment_recognizer is not None
                    and label in STATE.sentence_segment_recognizer.class_names
                ):
                    sentence_model_updated = _run_fast_bilstm_adaptation(
                        STATE.sentence_segment_recognizer, DEFAULT_SENTENCE_SEGMENT_CKPT
                    )

                if (
                    STATE.sentence_recognizer is not None
                    and label in STATE.sentence_recognizer.vocab.word_to_id
                ):
                    try:
                        ctc_vocab = STATE.sentence_recognizer.vocab
                        sent_cls = [w for k, w in sorted(ctc_vocab.id_to_word.items()) if k != 0]
                        clip_bank_212: Dict[str, List[np.ndarray]] = {c: [] for c in sent_cls}
                        for c in sent_cls:
                            w_dir = os.path.join(WEBCAM_NPY_DIR, c)
                            if os.path.isdir(w_dir):
                                for fn in sorted(os.listdir(w_dir))[-6:]:
                                    if fn.endswith(".npy"):
                                        try:
                                            wa = np.load(os.path.join(w_dir, fn), allow_pickle=False)
                                            w_core = extract_active_sign_windows_76x3(wa, fast_mode=True)[0]
                                            s212 = np.asarray(preprocess_npy_clip_for_ctc(w_core, target_clip_frames=16), dtype=np.float32)
                                            if len(s212) == 16:
                                                clip_bank_212[c].append(s212)
                                        except Exception:
                                            pass
                            if len(clip_bank_212[c]) < 2 and os.path.isdir(os.path.join(STATE.default_npy_root, "train")):
                                tr_root = os.path.join(STATE.default_npy_root, "train")
                                for fd in os.listdir(tr_root):
                                    if normalize_single_gloss(fd) == c:
                                        f_dir = os.path.join(tr_root, fd)
                                        for fn in sorted(os.listdir(f_dir))[:4]:
                                            if fn.endswith(".npy"):
                                                try:
                                                    va = np.load(os.path.join(f_dir, fn), allow_pickle=False)
                                                    s212 = np.asarray(preprocess_npy_clip_for_ctc(va, target_clip_frames=16), dtype=np.float32)
                                                    if len(s212) == 16:
                                                        clip_bank_212[c].append(s212)
                                                except Exception:
                                                    pass
                        ctc_seqs: List[np.ndarray] = []
                        ctc_targets: List[List[int]] = []
                        avail_c = [c for c in sent_cls if clip_bank_212[c]]
                        for c in avail_c:
                            cid = int(ctc_vocab.word_to_id[c])
                            for s16 in clip_bank_212[c]:
                                ctc_seqs.append(np.repeat(s16, 2, axis=0))  # 32 frames single-word anchor
                                ctc_targets.append([cid])
                        for c1 in avail_c:
                            for c2 in avail_c:
                                if c1 == c2:
                                    continue
                                id1, id2 = int(ctc_vocab.word_to_id[c1]), int(ctc_vocab.word_to_id[c2])
                                for k in range(3):
                                    a1 = clip_bank_212[c1][k % len(clip_bank_212[c1])]
                                    a2 = clip_bank_212[c2][k % len(clip_bank_212[c2])]
                                    mid = 0.5 * (a1[-1:] + a2[:1])
                                    ctc_seqs.append(np.concatenate([a1, mid, a2], axis=0))  # 33 frames 2-word sentence
                                    ctc_targets.append([id1, id2])
                        if len(ctc_seqs) >= 4:
                            dev = STATE.sentence_recognizer.device
                            c_mdl = STATE.sentence_recognizer.model
                            max_t = max(s.shape[0] for s in ctc_seqs)
                            x_pad = np.zeros((len(ctc_seqs), max_t, CTC_FEATURE_DIM), dtype=np.float32)
                            in_lens = []
                            flat_t = []
                            t_lens = []
                            for i, (s, tg) in enumerate(zip(ctc_seqs, ctc_targets)):
                                x_pad[i, : s.shape[0]] = s
                                in_lens.append(s.shape[0])
                                flat_t.extend(tg)
                                t_lens.append(len(tg))
                            x_ten = torch.tensor(x_pad, device=dev)
                            in_ten = torch.tensor(in_lens, dtype=torch.long, device=dev)
                            tg_ten = torch.tensor(flat_t, dtype=torch.long, device=dev)
                            tl_ten = torch.tensor(t_lens, dtype=torch.long, device=dev)
                            c_opt = torch.optim.AdamW(c_mdl.parameters(), lr=6e-4, weight_decay=1e-4)
                            c_crit = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
                            c_mdl.train()
                            for _ in range(10):
                                c_opt.zero_grad()
                                lp = c_mdl(x_ten, in_ten).transpose(0, 1)  # [T, B, C]
                                loss = c_crit(lp, tg_ten, in_ten, tl_ten)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(c_mdl.parameters(), 2.0)
                                c_opt.step()
                            c_mdl.eval()
                            if os.path.exists(DEFAULT_SENTENCE_CKPT):
                                sck = torch.load(DEFAULT_SENTENCE_CKPT, map_location="cpu", weights_only=False)
                                sck["model_state_dict"] = {k: v.detach().cpu() for k, v in c_mdl.state_dict().items()}
                                torch.save(sck, DEFAULT_SENTENCE_CKPT)
                            sentence_model_updated = True
                    except Exception as e:
                        print(f"[FINETUNE-CTC] error: {e}")

                if sentence_model_updated:
                    STATE._rebuild_sentence_spotting()

                pred = _predict_word_clip(clip_to_save)
                STATE.last_word_prediction = pred
                STATE.word_peak_prediction = pred
                STATE.word_ema_probs = None
                counts = STATE.get_webcam_sample_counts()

                self._send_json(
                    {
                        "ok": True,
                        "saved_npy_path": save_path,
                        "adapted_label": label,
                        "saved_count_for_label": counts.get(label, 1),
                        "webcam_sample_counts": counts,
                        "sentence_model_updated": sentence_model_updated,
                        "finetune_samples": int(len(xs)),
                        "verify_prediction": pred,
                    }
                )
                return

            if self.path == "/api/stream_reset":
                STATE.reset_word_stream()
                STATE.reset_sentence_stream()
                self._send_json({"ok": True})
                return

            self._send_json({"ok": False, "error": f"Unknown POST path {self.path}"}, status=404)
        except Exception as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                status=400,
            )


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), DualVSLRequestHandler)
    print(f"VSL Studio (.npy keypoints_splited enabled) running at http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
