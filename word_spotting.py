#!/usr/bin/env python3
"""
===============================================================================
WORD-SPOTTING SENTENCE DECODER — Nhận diện CÂU bằng mô hình TỪ ĐƠN (BiLSTM) + LM
===============================================================================
Ý tưởng "biến tấu" khi KHÔNG có dataset câu:
  - Mô hình Từ đơn BiLSTM (30x138) đã học rất tốt từng ký hiệu riêng lẻ (và đã được
    tinh chỉnh theo góc webcam của người dùng).
  - Một câu ký hiệu = chuỗi các ký hiệu từ đơn nối tiếp nhau + các đoạn chuyển tiếp.
  => Quét mọi đoạn cửa sổ ứng viên [s, e) trên clip câu (T, 76, 3), chấm điểm từng đoạn
     bằng mô hình Từ đơn, rồi dùng QUY HOẠCH ĐỘNG (Viterbi phân đoạn) kết hợp Mô hình
     Ngôn ngữ (bigram LM sinh từ ngữ pháp tiếng Việt) để tìm cách chia câu tốt nhất:

     D[e, c] = max_{s, c'} D[s, c'] + w_ac·L·log P(c | clip[s:e]) + α·log P_LM(c | c') + β
     D[e, c] = max(D[e, c], D[e - Δ, c] + γ·Δ)                     (bỏ qua frame chuyển tiếp)
     Kết thúc: max_c D[T, c] + α·log P_LM(</s> | c)

  - L = độ dài đoạn (frame) nên một ký hiệu đúng (P≈1) gần như không bị phạt, còn bỏ
    qua frame thì bị phạt γ/frame; β (âm) phạt mỗi từ để tránh chia vụn 1 ký hiệu.
  - Tối ưu tốc độ: Tiền tính toán đặc trưng 138-D cho từng frame chỉ 1 lần duy nhất O(T),
    sau đó nội suy tuyến tính NumPy cho mọi đoạn ứng viên [s, e) chỉ mất vài mili-giây.
===============================================================================
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from feature_extractor import SINGLE_SIGN_FEATURE_DIM, SINGLE_SIGN_SEQUENCE_LENGTH, Landmark3D
from vsl400_npy_loader import (
    _extract_bimanual_pose_32_context,
    _extract_hand_106_with_wrist,
    canonicalize_body_frame_76x3,
    interpolate_and_trim_npy_76x3,
    parse_npy_frame_76x3,
)


def _precompute_frame_features_138(arr_tx76x3: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Precompute per-frame 106-D hand features and 32-D pose/bimanual context in O(T)
    so any sub-segment `[s:e]` can be assembled and resampled to `(30, 138)` in microseconds.
    """
    arr = canonicalize_body_frame_76x3(arr_tx76x3)
    t_total = arr.shape[0]

    right_106 = np.zeros((t_total, 106), dtype=np.float32)
    left_106 = np.zeros((t_total, 106), dtype=np.float32)
    ctx_32 = np.zeros((t_total, 32), dtype=np.float32)
    has_right = np.zeros(t_total, dtype=np.int32)
    has_left = np.zeros(t_total, dtype=np.int32)

    prev_rw: Optional[Landmark3D] = None
    prev_lw: Optional[Landmark3D] = None

    for t in range(t_total):
        pose_pts, left_pts, right_pts = parse_npy_frame_76x3(arr[t])
        hr = len(right_pts) == 21
        hl = len(left_pts) == 21
        has_right[t] = 1 if hr else 0
        has_left[t] = 1 if hl else 0

        if hr:
            v_cont, cur_rw = _extract_hand_106_with_wrist(right_pts, prev_rw)
            right_106[t] = v_cont
        else:
            cur_rw = None

        if hl:
            v_cont_l, cur_lw = _extract_hand_106_with_wrist(left_pts, prev_lw)
            left_106[t] = v_cont_l
        else:
            cur_lw = None

        ctx_32[t] = _extract_bimanual_pose_32_context(
            pose_pts=pose_pts,
            left_pts=left_pts,
            right_pts=right_pts,
            prev_left_wrist=prev_lw,
            prev_right_wrist=prev_rw,
        )

        prev_rw = cur_rw
        prev_lw = cur_lw

    # Zero out the frame-to-frame velocity slice for the first frame of any segment
    right_106_first = right_106.copy()
    right_106_first[:, 102:106] = 0.0
    left_106_first = left_106.copy()
    left_106_first[:, 102:106] = 0.0
    ctx_32_first = ctx_32.copy()
    ctx_32_first[:, 16:22] = 0.0

    # Pre-combine right-primary and left-primary 138-D tracks (with fallback when primary hand is missing)
    primary_r_cont = np.where(has_right[:, None] == 1, right_106, left_106_first)
    primary_r_first = np.where(has_right[:, None] == 1, right_106_first, left_106_first)
    primary_l_cont = np.where(has_left[:, None] == 1, left_106, right_106_first)
    primary_l_first = np.where(has_left[:, None] == 1, left_106_first, right_106_first)

    r_cont_138 = np.concatenate([primary_r_cont, ctx_32], axis=1)
    l_cont_138 = np.concatenate([primary_l_cont, ctx_32], axis=1)

    # Compute Kinematic Abrupt-Transition / Change-Point Bonus B[t] (length T + 1)
    # Detects sudden handshape changes, velocity direction reversals, or body-zone shifts
    # when the signer switches signs abruptly WITHOUT lowering hands.
    boundary_bonus = np.zeros(t_total + 1, dtype=np.float32)
    if t_total >= 4:
        # Handshape + Pose NMM context change between t-1 and t
        feat_diff = np.linalg.norm(r_cont_138[1:, :63] - r_cont_138[:-1, :63], axis=1)
        ctx_diff = np.linalg.norm(ctx_32[1:, :12] - ctx_32[:-1, :12], axis=1)
        # Wrist velocity vectors (dx, dy, dz) in ctx_32[12:15] (left) and ctx_32[16:19] (right)
        v_r = ctx_32[:, 16:19]
        v_l = ctx_32[:, 12:15]
        v = np.where(has_right[:, None] == 1, v_r, v_l)
        v_norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
        v_unit = v / v_norm
        # Direction change (1 - cos(theta)) between frame t-1 and t
        dir_flip = 1.0 - np.sum(v_unit[1:] * v_unit[:-1], axis=1)
        raw_cp = 0.45 * feat_diff + 0.80 * ctx_diff + 0.35 * dir_flip
        std_cp = float(np.std(raw_cp)) if raw_cp.size > 0 else 1.0
        if std_cp > 1e-5:
            # Zero-centered so it favors true abrupt sign boundaries without inflating segment count
            boundary_bonus[1:t_total] = np.clip((raw_cp - float(np.mean(raw_cp))) / std_cp, -1.0, 1.5)

    return {
        "r_cont": r_cont_138,
        "r_first": np.concatenate([primary_r_first, ctx_32_first], axis=1),
        "l_cont": l_cont_138,
        "l_first": np.concatenate([primary_l_first, ctx_32_first], axis=1),
        "has_right_cumsum": np.concatenate([[0], np.cumsum(has_right)]),
        "has_left_cumsum": np.concatenate([[0], np.cumsum(has_left)]),
        "boundary_bonus": boundary_bonus,
    }


def _slice_and_resample_138(
    pre: Dict[str, np.ndarray],
    s: int,
    e: int,
    target_len: int = SINGLE_SIGN_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Slice `[s:e]` from precomputed 138-D tracks and linearly interpolate to `(30, 138)`."""
    n_right = int(pre["has_right_cumsum"][e] - pre["has_right_cumsum"][s])
    n_left = int(pre["has_left_cumsum"][e] - pre["has_left_cumsum"][s])
    use_right = n_right >= max(1, int(0.3 * n_left))

    cont = pre["r_cont"] if use_right else pre["l_cont"]
    first = pre["r_first"] if use_right else pre["l_first"]

    seg = cont[s:e].copy()
    seg[0] = first[s]

    # Filter all-zero rows if non-empty rows exist
    non_zero_mask = np.any(np.abs(seg) > 1e-6, axis=1)
    if int(np.sum(non_zero_mask)) >= 2:
        seg = seg[non_zero_mask]

    m = seg.shape[0]
    if m > 1:
        # Normalize frame-to-frame velocity magnitude to the ~34-frame reference training duration
        # so fast signs (e.g. 8-15 frames) have the exact same velocity distribution as 34-frame signs.
        vel_scale = np.float32(np.clip(float(m) / 34.0, 0.25, 1.5))
        seg[1:, 102:106] *= vel_scale
        seg[1:, 122:128] *= vel_scale

    if m == target_len:
        return seg
    if m == 1:
        return np.repeat(seg, target_len, axis=0)

    pos = np.linspace(0.0, float(m - 1), target_len, dtype=np.float32)
    lo = np.floor(pos).astype(np.int32)
    hi = np.minimum(m - 1, lo + 1)
    frac = (pos - lo)[:, None].astype(np.float32)
    return ((np.float32(1.0) - frac) * seg[lo] + frac * seg[hi]).astype(np.float32)


class WordSpottingSentenceDecoder:
    def __init__(
        self,
        word_recognizer: Any,
        lm: Optional[Any] = None,
        pos_step: int = 3,
        min_sign_frames: int = 6,
        max_sign_frames: int = 66,
        length_step: int = 4,
        acoustic_weight: float = 0.20,
        lm_alpha: float = 1.5,
        word_penalty: float = -1.0,
        skip_penalty_per_frame: float = -0.15,
        max_words: int = 8,
        ctc_recognizer: Optional[Any] = None,
    ) -> None:
        self.rec = word_recognizer
        self.class_names: List[str] = list(word_recognizer.class_names)
        self.lm = lm
        self.pos_step = int(pos_step)
        self.min_sign_frames = int(min_sign_frames)
        self.max_sign_frames = int(max_sign_frames)
        self.length_step = int(length_step)
        self.acoustic_weight = float(acoustic_weight)
        self.lm_alpha = float(lm_alpha)
        self.word_penalty = float(word_penalty)
        self.skip_penalty_per_frame = float(skip_penalty_per_frame)
        self.max_words = int(max_words)
        self.ctc_recognizer = ctc_recognizer

    # ------------------------------------------------------------------
    def _lm_logp(self, prev: Optional[str], cur: str) -> float:
        if prev is not None and prev == cur:
            return -6.0
        if self.lm is None or self.lm_alpha == 0.0:
            return 0.0
        raw_lp = float(self.lm.log_prob(prev, cur))
        if cur == "</s>":
            return raw_lp
        # Shift by log(K) (log-likelihood ratio vs uniform 1/K) so adding words in fast sentences
        # is not penalized by -log(K) per word.
        return raw_lp + math.log(max(2, len(self.class_names)))

    def _adaptive_lengths(self, t_total: int) -> List[int]:
        if t_total < 16:
            return sorted({max(4, t_total // 2), t_total})
        # Scale candidate sign window lengths proportionally to total sentence duration `t_total`
        # so fast 2-3x sentences (e.g. T=30..65 frames for 2-3 words) segment into 2-4 words
        # instead of being swallowed by a single 66-frame window.
        lo = max(5, min(self.min_sign_frames, int(round(t_total * 0.14))))
        hi = min(self.max_sign_frames, max(12, int(round(t_total * 0.56))))
        step = max(2, min(self.length_step, max(2, (hi - lo) // 5)))
        lengths = list(range(lo, hi + 1, step))
        if hi not in lengths:
            lengths.append(hi)
        return sorted(set(lengths))

    def _score_segments(
        self, arr: np.ndarray, grid: List[int]
    ) -> Dict[Tuple[int, int], np.ndarray]:
        t_total = arr.shape[0]
        lengths = set(self._adaptive_lengths(t_total))
        max_len = max(lengths) if lengths else self.max_sign_frames
        min_len = min(lengths) if lengths else self.min_sign_frames
        pairs: List[Tuple[int, int]] = []
        for si, s in enumerate(grid):
            for ei in range(si + 1, len(grid)):
                e = grid[ei]
                L = e - s
                if L > max_len + self.pos_step:
                    break
                if L < min_len or not any(abs(L - l) <= self.pos_step // 2 + 1 for l in lengths):
                    continue
                pairs.append((si, ei))
        if not pairs:
            return {}

        pre = _precompute_frame_features_138(arr)
        batch_np = np.stack(
            [_slice_and_resample_138(pre, grid[si], grid[ei]) for si, ei in pairs],
            axis=0,
        )
        x = torch.from_numpy(batch_np).to(self.rec.device)
        probs_all: List[np.ndarray] = []
        with torch.no_grad():
            for b in range(0, x.shape[0], 256):
                probs_all.append(self.rec.model(x[b : b + 256]).cpu().numpy())
        probs = np.concatenate(probs_all, axis=0)
        self._last_boundary_bonus = pre.get("boundary_bonus")
        out_probs: Dict[Tuple[int, int], np.ndarray] = {pair: probs[k].copy() for k, pair in enumerate(pairs)}

        # Suppress composite multi-word spans: if a long span (si, ei) can be split at `mi`
        # into two sub-spans (si, mi) and (mi, ei) that confidently predict TWO DIFFERENT words,
        # prevent Softmax on the merged span from swallowing the shorter word (e.g. TÔI + ĐAU -> ĐAU).
        for si, ei in pairs:
            L = grid[ei] - grid[si]
            if L < 22:
                continue
            for mi in range(si + 1, ei):
                p_l = out_probs.get((si, mi))
                p_r = out_probs.get((mi, ei))
                if p_l is None or p_r is None:
                    continue
                c_l = int(np.argmax(p_l))
                c_r = int(np.argmax(p_r))
                if c_l != c_r and float(p_l[c_l]) >= 0.65 and float(p_r[c_r]) >= 0.65:
                    out_probs[(si, ei)] = np.clip(out_probs[(si, ei)] * 0.12, 1e-5, 1.0)
                    break

        return out_probs

    # ------------------------------------------------------------------
    def decode(self, clip_76x3: np.ndarray, top_k_segment: int = 3) -> Dict[str, Any]:
        t0 = time.perf_counter()
        arr = interpolate_and_trim_npy_76x3(np.asarray(clip_76x3, dtype=np.float32))
        t_total = int(arr.shape[0])
        if t_total < 6:
            raise ValueError("Clip câu quá ngắn (cần >= 6 frame có bàn tay).")

        grid = list(range(0, t_total, self.pos_step))
        if grid[-1] != t_total:
            grid.append(t_total)
        n_pos = len(grid)

        seg_probs = self._score_segments(arr, grid)
        n_cls = len(self.class_names)
        log_eps = 1e-6

        # States: index 0 = <s> (chưa có từ), 1..n_cls = từ cuối cùng
        NEG = -1e18
        D = np.full((n_pos, n_cls + 1), NEG, dtype=np.float64)
        n_words = np.zeros((n_pos, n_cls + 1), dtype=np.int32)
        back: Dict[Tuple[int, int], Tuple[int, int, str]] = {}
        D[0, 0] = 0.0

        by_end: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for (si, ei), p in seg_probs.items():
            by_end.setdefault(ei, []).append((si, p))

        lm_table = np.zeros((n_cls + 1, n_cls + 1), dtype=np.float64)
        for ps in range(n_cls + 1):
            prev_g = None if ps == 0 else self.class_names[ps - 1]
            for c in range(n_cls):
                lm_table[ps, c + 1] = self.lm_alpha * self._lm_logp(prev_g, self.class_names[c])

        for ei in range(1, n_pos):
            gap = grid[ei] - grid[ei - 1]
            skip_cost = self.skip_penalty_per_frame * gap
            for st in range(n_cls + 1):
                cand = D[ei - 1, st] + skip_cost
                if cand > D[ei, st]:
                    D[ei, st] = cand
                    n_words[ei, st] = n_words[ei - 1, st]
                    back[(ei, st)] = (ei - 1, st, "skip")
            for si, p in by_end.get(ei, []):
                L = grid[ei] - grid[si]
                cp_bonus = 0.0
                if getattr(self, "_last_boundary_bonus", None) is not None:
                    b_arr = self._last_boundary_bonus
                    cp_bonus = 0.12 * float(b_arr[grid[si]] + b_arr[grid[ei]])
                ac = (self.acoustic_weight * L + 1.2) * np.log(np.maximum(p, log_eps)) + cp_bonus
                valid_ps = np.where((D[si] > NEG / 2) & (n_words[si] < self.max_words))[0]
                if valid_ps.size == 0:
                    continue
                # mat shape: [len(valid_ps), n_cls]
                mat = D[si, valid_ps][:, None] + lm_table[valid_ps, 1:] + ac[None, :] + self.word_penalty
                best_idx = np.argmax(mat, axis=0)
                best_scores = mat[best_idx, np.arange(n_cls)]
                best_ps = valid_ps[best_idx]
                for c in range(n_cls):
                    st = c + 1
                    if best_scores[c] > D[ei, st]:
                        ps = int(best_ps[c])
                        D[ei, st] = float(best_scores[c])
                        n_words[ei, st] = int(n_words[si, ps]) + 1
                        back[(ei, st)] = (si, ps, "word")

        best_score, best_state = NEG, -1
        for st in range(1, n_cls + 1):
            if D[n_pos - 1, st] <= NEG / 2:
                continue
            end_lm = self.lm_alpha * self._lm_logp(self.class_names[st - 1], "</s>")
            sc = D[n_pos - 1, st] + end_lm
            if sc > best_score:
                best_score, best_state = sc, st

        segments: List[Dict[str, Any]] = []
        if best_state > 0:
            ei, st = n_pos - 1, best_state
            while ei > 0:
                pi, ps, kind = back[(ei, st)]
                if kind == "word":
                    p = seg_probs[(pi, ei)]
                    order = np.argsort(p)[::-1][:top_k_segment]
                    segments.append(
                        {
                            "gloss": self.class_names[st - 1],
                            "start_frame": int(grid[pi]),
                            "end_frame": int(grid[ei]),
                            "confidence": float(p[st - 1]),
                            "top3": [{"label": self.class_names[int(k)], "prob": float(p[int(k)])} for k in order],
                        }
                    )
                ei, st = pi, ps
            segments.reverse()

        glosses = [s["gloss"] for s in segments]
        conf = float(np.exp(np.mean([math.log(max(s["confidence"], 1e-6)) for s in segments]))) if segments else 0.0

        # Hybrid Fast-Sign CTC Fallback: only consult CTC when the 138-D BiLSTM segment decoder
        # is uncertain (`conf < 0.68`) so high-confidence BiLSTM segment predictions (which have
        # full 32-D body/shoulder context) are never overwritten by 212-D CTC errors.
        if self.ctc_recognizer is not None and t_total >= 12 and conf < 0.68:
            ctc_info = _fast_ctc_decode_for_fusion(self.ctc_recognizer, arr, beam_width=8)
            ctc_glosses = ctc_info["predicted_glosses"]
            ctc_conf = ctc_info["ac_conf"]
            use_ctc = False
            if len(ctc_glosses) >= 2 and ctc_conf > conf + 0.12:
                use_ctc = True
            if use_ctc:
                glosses = ctc_glosses
                conf = float(max(conf, ctc_conf))
                k_words = len(glosses)
                segments = [
                    {
                        "gloss": g,
                        "start_frame": int(round(idx * t_total / k_words)),
                        "end_frame": int(round((idx + 1) * t_total / k_words)),
                        "confidence": conf,
                        "top3": [{"label": g, "prob": conf}],
                    }
                    for idx, g in enumerate(glosses)
                ]

        return {
            "predicted_glosses": glosses,
            "segments": segments,
            "score": float(best_score) if best_state > 0 else None,
            "confidence": conf,
            "lm_log_prob": float(self.lm.sentence_log_prob(glosses)) if (self.lm is not None and glosses) else None,
            "num_candidate_segments": len(seg_probs),
            "clip_frames": t_total,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "engine": "word_spotting",
        }


def _fast_ctc_decode_for_fusion(ctc_recognizer: Any, arr_tx76x3: np.ndarray, beam_width: int = 8) -> Dict[str, Any]:
    """Fast 2-scale CTC decode that normalizes fast/slow clips into CTC's [32..58] frame sweet spot."""
    from vsl400_npy_loader import preprocess_npy_clip_for_ctc

    n = int(arr_tx76x3.shape[0])
    if n < 48:
        targets = sorted({max(28, min(48, int(round(n * 0.85)))), max(34, min(52, int(round(n * 1.05))))})
    elif n < 80:
        targets = sorted({max(32, min(54, int(round(n * 0.65)))), max(36, min(58, int(round(n * 0.78))))})
    else:
        targets = sorted({max(36, min(58, int(round(n * 0.48)))), max(40, min(64, int(round(n * 0.60))))})

    best_glosses: List[str] = []
    best_ac_conf = 0.0
    best_score = -1e9
    for tl in targets:
        seq212 = preprocess_npy_clip_for_ctc(arr_tx76x3, target_clip_frames=tl)
        if len(seq212) < 4:
            continue
        res = ctc_recognizer.predict_from_features(seq212, decode_method="beam_search_lm", beam_width=beam_width)
        raw_g = list(res.get("predicted_glosses", []))
        g_list = [g for i, g in enumerate(raw_g) if i == 0 or g != raw_g[i - 1]]
        probs = res.get("frame_probabilities")
        if probs is not None and len(probs) > 0:
            non_blank_max = np.max(probs[:, 1:], axis=1) if probs.shape[1] > 1 else np.max(probs, axis=1)
            top_k = np.sort(non_blank_max)[-max(4, len(non_blank_max) // 3) :]
            ac_conf = float(np.mean(top_k))
        else:
            ac_conf = 0.5
        w_bonus = 0.12 * min(len(g_list), 4) if len(g_list) >= 2 else (-0.25 if len(g_list) == 1 else -1.0)
        sc = ac_conf + w_bonus
        if sc > best_score:
            best_score = sc
            best_ac_conf = ac_conf
            best_glosses = g_list
    return {"predicted_glosses": best_glosses, "ac_conf": best_ac_conf, "score": best_score}


def concat_raw_clips_76x3(clips: Sequence[np.ndarray], transition_frames: int = 1) -> np.ndarray:
    """Nối nhiều clip từ đơn `(T_i, 76, 3)` (đã trim & chuẩn hoá khung vai) thành 1 câu liên tục chuyển ký hiệu đột ngột."""
    out: List[np.ndarray] = []
    for k, clip in enumerate(clips):
        c = canonicalize_body_frame_76x3(
            interpolate_and_trim_npy_76x3(np.asarray(clip, dtype=np.float32))
        )
        if k > 0 and out and transition_frames > 0:
            a, b = out[-1][-1], c[0]
            both = (np.abs(a) > 1e-6) & (np.abs(b) > 1e-6)
            for step in range(1, transition_frames + 1):
                alpha = step / float(transition_frames + 1)
                f = np.where(both, (1.0 - alpha) * a + alpha * b, b if alpha >= 0.5 else a)
                out.append(f[None].astype(np.float32))
        out.append(c)
    return np.concatenate(out, axis=0)


class ContinuousAbruptStreamTracker:
    """
    Bộ giải mã câu liên tục thời gian thực (Streaming) chuyên trị ký hiệu NHANH và chuyển ký hiệu
    đột ngột MÀ KHÔNG CẦN HẠ TAY XUỐNG RESET.

    Thiết kế "Utterance-level" (KHÔNG dùng sliding buffer cắt cửa sổ):
      - Người ký hiệu làm một câu rất nhanh (mỗi từ chỉ ~6–12 frame) nên không có thời gian để
        đối chiếu bằng cửa sổ trượt. Thay vào đó tracker GIỮ TOÀN BỘ khung hình của câu đang ký
        (từ lúc giơ tay đến lúc hạ tay / dừng tay) và chạy Quy hoạch động Phân đoạn trên cả câu:
        DP tự tìm điểm ranh giới tối ưu giữa các từ (kể cả từ chỉ dài 6 frame).
      - Nhận frame theo LÔ (`push_frames`) để không rơi frame nào khi mạng/AI đang bận:
        mọi frame đều được lưu, nhưng mỗi lần gọi chỉ giải mã TỐI ĐA 1 lần (xem trước trực tiếp).
      - Khi hạ tay hoặc dừng tay → chạy 1 lần GIẢI MÃ CHỐT CÂU với lưới mịn hơn trên toàn bộ câu
        (chính xác nhất), rồi sẵn sàng cho câu tiếp theo.
      - Câu rất dài (> 300 frame không hạ tay) mới khoá bớt các từ đầu để giữ độ trễ thấp.
    """

    def __init__(
        self,
        segment_recognizer: Any,
        lm: Optional[Any] = None,
        lm_weight: float = 0.22,
        commit_conf_threshold: float = 0.52,
        min_new_frames_for_preview: int = 5,
        end_missing_frames: int = 4,
        end_still_frames: int = 26,
        ctc_recognizer: Optional[Any] = None,
    ) -> None:
        self.rec = segment_recognizer
        self.class_names: List[str] = list(segment_recognizer.class_names)
        self.lm = lm
        self.ctc_recognizer = ctc_recognizer
        self.min_new_frames_for_preview = int(min_new_frames_for_preview)
        self.end_missing_frames = int(end_missing_frames)
        self.end_still_frames = int(end_still_frames)
        # Live preview decoder: coarse grid, allows very short (fast) signs >= 6 frames
        self.decoder = WordSpottingSentenceDecoder(
            segment_recognizer,
            lm=lm,
            pos_step=4,
            min_sign_frames=6,
            length_step=5,
            skip_penalty_per_frame=-0.22,
            ctc_recognizer=ctc_recognizer,
        )
        # Final decoder (runs once per sentence when hands go down): fine grid + CTC hybrid fusion
        self.final_decoder = WordSpottingSentenceDecoder(
            segment_recognizer,
            lm=lm,
            pos_step=3,
            min_sign_frames=6,
            length_step=3,
            skip_penalty_per_frame=-0.22,
            ctc_recognizer=ctc_recognizer,
        )
        self.reset()

    def reset(self) -> None:
        self.active_frames: List[np.ndarray] = []
        self.total_frame_idx: int = 0
        self.buffer_offset_idx: int = 0
        self.missing_hand_streak: int = 0
        self.still_hand_streak: int = 0
        self.sentence_completed: bool = False
        self.committed_segments: List[Dict[str, Any]] = []
        self.live_tail_segments: List[Dict[str, Any]] = []
        self.prev_wrist_vec: Optional[np.ndarray] = None
        self.frames_since_decode: int = 0
        self.signing_frames: int = 0
        self.last_decode_ms: float = 0.0

    def _active_wrist(self, frame_76x3: np.ndarray) -> Optional[np.ndarray]:
        r_w = frame_76x3[54]
        if np.any(np.abs(r_w) > 1e-5):
            return r_w.copy()
        l_w = frame_76x3[33]
        if np.any(np.abs(l_w) > 1e-5):
            return l_w.copy()
        return None

    def _start_new_sentence(self) -> None:
        self.committed_segments = []
        self.live_tail_segments = []
        self.active_frames = []
        self.sentence_completed = False
        self.signing_frames = 0
        self.frames_since_decode = 0
        self.buffer_offset_idx = self.total_frame_idx - 1

    def _ingest(self, frame_76x3: np.ndarray, hand_count: int) -> bool:
        """Cheap per-frame state update (no AI). Returns True if the sentence just reached a pause/end."""
        self.total_frame_idx += 1

        wrist = self._active_wrist(frame_76x3) if hand_count > 0 else None
        wrist_speed = 0.0
        if wrist is not None:
            if self.prev_wrist_vec is not None:
                wrist_speed = float(np.linalg.norm((wrist - self.prev_wrist_vec)[:2]))
            self.prev_wrist_vec = wrist
            self.still_hand_streak = self.still_hand_streak + 1 if wrist_speed < 0.0045 else 0
        else:
            self.prev_wrist_vec = None

        if hand_count > 0:
            # Only start a brand-new sentence if hands were lowered for >= 10 frames or held still >= 28 frames.
            # If it was just a short 3..9 frame fast-motion dropout, resume the SAME sentence without losing words!
            if self.sentence_completed:
                if self.missing_hand_streak >= 10 or (self.still_hand_streak == 0 and wrist_speed > 0.012 and len(self.active_frames) == 0):
                    self._start_new_sentence()
                else:
                    self.sentence_completed = False
            self.missing_hand_streak = 0
            if not self.active_frames:
                self.buffer_offset_idx = self.total_frame_idx - 1
            self.active_frames.append(frame_76x3)
            self.frames_since_decode += 1
            self.signing_frames += 1
        else:
            self.missing_hand_streak += 1
            # Keep short fast-motion dropouts (<= 6 frames) inside active_frames so interpolation bridges them
            if 0 < len(self.active_frames) and self.missing_hand_streak <= 6:
                self.active_frames.append(frame_76x3)
                self.frames_since_decode += 1
            elif self.sentence_completed and self.missing_hand_streak >= 10:
                # Hands stayed down for >= 10 frames: clear active_frames so next hand raise starts a new sentence
                self.active_frames = []

        if self.sentence_completed or self.signing_frames < 6:
            return False
        return (
            self.missing_hand_streak == self.end_missing_frames
            or self.still_hand_streak == self.end_still_frames
        )

    def _run_decode(self, final: bool) -> None:
        if len(self.active_frames) < 6:
            return
        t0 = time.perf_counter()
        clip_np = np.stack(self.active_frames, axis=0)
        dec = self.final_decoder if final else self.decoder
        try:
            ws_res = dec.decode(clip_np)
        except ValueError:
            return
        self.last_decode_ms = (time.perf_counter() - t0) * 1000.0
        self.frames_since_decode = 0
        mapped_segs: List[Dict[str, Any]] = [
            {
                "gloss": str(s["gloss"]),
                "start_frame": int(self.buffer_offset_idx + s["start_frame"]),
                "end_frame": int(self.buffer_offset_idx + s["end_frame"]),
                "confidence": float(s["confidence"]),
                "_local_end": int(s["end_frame"]),
            }
            for s in ws_res.get("segments", [])
        ]

        # Prefix locking only for very long continuous signing (> 300 frames without lowering hands)
        while not final and len(self.active_frames) > 300 and len(mapped_segs) >= 2:
            first_seg = mapped_segs[0]
            cut = int(first_seg["_local_end"])
            if 0 < cut <= len(self.active_frames) - 60:
                if not self.committed_segments or self.committed_segments[-1]["gloss"] != first_seg["gloss"]:
                    self.committed_segments.append({k: v for k, v in first_seg.items() if k != "_local_end"})
                self.active_frames = self.active_frames[cut:]
                self.buffer_offset_idx += cut
                mapped_segs.pop(0)
                for rem in mapped_segs:
                    rem["_local_end"] -= cut
            else:
                break

        self.live_tail_segments = [{k: v for k, v in s.items() if k != "_local_end"} for s in mapped_segs]

    def push_frames(self, frames: Sequence[np.ndarray], hand_counts: Sequence[int]) -> Dict[str, Any]:
        """
        Ingest a BATCH of frames (every frame captured by the browser, none dropped) and run
        at most ONE decode: a fine-grid final decode if the sentence paused/ended inside this batch,
        otherwise a coarse live-preview decode when enough new frames arrived.
        """
        t0 = time.perf_counter()
        ended = False
        for f, h in zip(frames, hand_counts):
            if self._ingest(f, int(h)):
                ended = True
        if ended:
            self._run_decode(final=True)
            if self.live_tail_segments or self.committed_segments:
                self.sentence_completed = True
        elif not self.sentence_completed:
            if self.frames_since_decode >= self.min_new_frames_for_preview and len(self.active_frames) >= 8:
                self._run_decode(final=False)

        display_segments: List[Dict[str, Any]] = list(self.committed_segments)
        for s in self.live_tail_segments:
            if not display_segments or display_segments[-1]["gloss"] != s["gloss"]:
                display_segments.append(s)

        glosses = [s["gloss"] for s in display_segments]
        conf = float(np.mean([s["confidence"] for s in display_segments])) if display_segments else 0.0
        return {
            "predicted_glosses": glosses,
            "segments": display_segments,
            "confidence": conf,
            "active_window_frames": len(self.active_frames),
            "committed_words": len(display_segments),
            "sentence_completed": self.sentence_completed,
            "decode_ms": float(self.last_decode_ms),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "engine": "word_spotting",
        }

    def step(self, frame_76x3: np.ndarray, hand_count: int) -> Dict[str, Any]:
        return self.push_frames([frame_76x3], [hand_count])



def build_raw_sentence_clips_from_npy_root(
    root_dir: str,
    class_names: Sequence[str],
    num_sentences: int = 20,
    split: str = "test",
    seed: int = 0,
    use_grammar: bool = True,
    max_files_per_class: int = 40,
    transition_frames: int = 1,
) -> List[Dict[str, Any]]:
    """
    Ghép câu RAW `(T, 76, 3)` từ các file `.npy` từ đơn trong `root_dir/<split>/<Từ>/*.npy`
    (mặc định tập test => không rò rỉ dữ liệu train) theo ngữ pháp `SentenceGrammar`.
    Trả về list `{"glosses": [...], "clip": np.ndarray, "files": [...]}`.
    """
    import os
    import random

    from decoder import normalize_single_gloss
    from sentence_grammar import SentenceGrammar

    rng = random.Random(seed)
    split_dir = os.path.join(root_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục {split_dir}")

    want = {normalize_single_gloss(c) for c in class_names}
    files_by_gloss: Dict[str, List[str]] = {}
    for folder in os.listdir(split_dir):
        g = normalize_single_gloss(folder)
        if g not in want:
            continue
        fdir = os.path.join(split_dir, folder)
        fs = sorted(os.path.join(fdir, f) for f in os.listdir(fdir) if f.endswith(".npy"))[:max_files_per_class]
        if fs:
            files_by_gloss[g] = fs
    avail = sorted(files_by_gloss.keys())
    if not avail:
        raise ValueError("Không có file .npy nào khớp với các lớp từ của mô hình trong tập " + split)

    grammar = SentenceGrammar(avail) if use_grammar else None
    if grammar is not None and not grammar.is_usable:
        grammar = None

    results: List[Dict[str, Any]] = []
    for _ in range(num_sentences):
        if grammar is not None:
            glosses = grammar.sample(rng)
        else:
            glosses = []
            for _ in range(rng.randint(2, min(3, max(2, len(avail))))):
                choices = [w for w in avail if not glosses or w != glosses[-1]] or avail
                glosses.append(rng.choice(choices))
        files = [rng.choice(files_by_gloss[g]) for g in glosses]
        clips = [np.load(f, allow_pickle=False) for f in files]
        results.append(
            {
                "glosses": glosses,
                "clip": concat_raw_clips_76x3(clips, transition_frames=transition_frames),
                "files": files,
            }
        )
    return results


__all__ = [
    "WordSpottingSentenceDecoder",
    "ContinuousAbruptStreamTracker",
    "concat_raw_clips_76x3",
    "build_raw_sentence_clips_from_npy_root",
]
