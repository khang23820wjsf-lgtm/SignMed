"""
Visualization utilities for VSL Continuous Recognition & Single-Sign models.
Generates publication-quality Confusion Matrix heatmaps, per-class metric charts,
and base64 image streams for immediate interactive rendering in the web interface.
"""

from __future__ import annotations

import base64
import io
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # Headless backend for web server / CLI execution
import matplotlib.pyplot as plt
import numpy as np


def align_token_sequences(
    ref: Sequence[str], hyp: Sequence[str]
) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Backtracked token-level Levenshtein alignment between reference and hypothesis glosses.
    Returns pairs of `(ref_token, hyp_token)`:
      - `(word, word)`: Hit (correct recognition)
      - `(word_a, word_b)`: Substitution (word_a misrecognized as word_b)
      - `(word_a, None)`: Deletion (omitted word)
      - `(None, word_b)`: Insertion (spurious word)
    """
    ref_list = list(ref or [])
    hyp_list = list(hyp or [])
    n = len(ref_list)
    m = len(hyp_list)

    if n == 0:
        return [(None, h) for h in hyp_list]
    if m == 0:
        return [(r, None) for r in ref_list]

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_list[i - 1] == hyp_list[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # Substitution
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                )

    i, j = n, m
    aligned: List[Tuple[Optional[str], Optional[str]]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_list[i - 1] == hyp_list[j - 1]:
            aligned.append((ref_list[i - 1], hyp_list[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned.append((ref_list[i - 1], hyp_list[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((ref_list[i - 1], None))
            i -= 1
        else:
            aligned.append((None, hyp_list[j - 1]))
            j -= 1

    aligned.reverse()
    return aligned


def compute_ctc_token_confusion_matrix(
    predictions: Sequence[Dict[str, Any]],
    vocab_words: Sequence[str],
    include_eps: bool = False,
) -> Dict[str, Any]:
    """
    Compute token-level confusion matrix from continuous sentence CTC predictions.
    Maps aligned `(ref_word, hyp_word)` pairs into a `[V, V]` confusion matrix.
    """
    clean_vocab = [w for w in vocab_words if w and w != "<blank>"]
    class_names = list(clean_vocab)
    if include_eps:
        class_names.append("<del/ins>")

    idx_map = {w: i for i, w in enumerate(class_names)}
    n_cls = len(class_names)
    matrix = np.zeros((n_cls, n_cls), dtype=np.int64)

    total_tokens = 0
    correct_tokens = 0

    for sample in predictions:
        ref = sample.get("reference") or []
        hyp = sample.get("hypothesis") or []
        pairs = align_token_sequences(ref, hyp)

        for r_tok, h_tok in pairs:
            if r_tok is not None and h_tok is not None:
                total_tokens += 1
                if r_tok == h_tok:
                    correct_tokens += 1
                if r_tok in idx_map and h_tok in idx_map:
                    matrix[idx_map[r_tok], idx_map[h_tok]] += 1
            elif include_eps:
                eps_idx = idx_map["<del/ins>"]
                if r_tok is not None and r_tok in idx_map:
                    matrix[idx_map[r_tok], eps_idx] += 1
                    total_tokens += 1
                elif h_tok is not None and h_tok in idx_map:
                    matrix[eps_idx, idx_map[h_tok]] += 1

    acc = float(correct_tokens) / float(max(1, total_tokens))
    return {
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
        "token_accuracy": acc,
        "total_tokens": total_tokens,
    }


def plot_confusion_matrix(
    matrix: Union[np.ndarray, Sequence[Sequence[int]], Sequence[Sequence[float]]],
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
    output_path: Optional[str] = None,
    normalize: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    cmap_name: str = "Blues",
) -> Dict[str, Any]:
    """
    Render a high-resolution Confusion Matrix heatmap.

    Returns:
      dict with:
        - "image_base64": Data URL string `data:image/png;base64,...`
        - "image_path": Path to saved PNG file (if `output_path` is given)
        - "confusion_matrix": Raw integer count matrix
        - "normalized_matrix": Per-row normalized percentage matrix
        - "class_names": List of labels
    """
    mat = np.asarray(matrix, dtype=np.float64)
    n_classes = mat.shape[0]

    if n_classes != len(class_names):
        class_names = [class_names[i] if i < len(class_names) else f"C{i}" for i in range(n_classes)]

    # Compute row sums for recall normalization
    row_sums = mat.sum(axis=1, keepdims=True)
    norm_mat = np.divide(
        mat,
        row_sums,
        out=np.zeros_like(mat, dtype=np.float64),
        where=(row_sums > 0),
    )

    display_mat = norm_mat if normalize else mat

    # Adaptive figure sizing
    if figsize is None:
        size = max(5.5, min(14.0, 1.2 + 0.75 * n_classes))
        figsize = (size + 1.2, size)

    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")

    cmap = plt.get_cmap(cmap_name)
    im = ax.imshow(display_mat, interpolation="nearest", cmap=cmap)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#cbd5e1", labelsize=9)
    cbar.set_label("Tỉ lệ (Recall)" if normalize else "Số lượng mẫu (Count)", color="#cbd5e1", fontsize=10)

    # Tick marks & labels
    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    # Clean label formatting (truncate if very long)
    clean_labels = [
        (name[:16] + "…") if len(name) > 17 else name for name in class_names
    ]
    ax.set_xticklabels(clean_labels, rotation=35, ha="right", color="#f8fafc", fontsize=max(8, min(11, 14 - int(n_classes * 0.4))))
    ax.set_yticklabels(clean_labels, color="#f8fafc", fontsize=max(8, min(11, 14 - int(n_classes * 0.4))))

    # Value text annotations inside cells
    thresh = (display_mat.max() + display_mat.min()) / 2.0 if display_mat.size > 0 else 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            val = mat[i, j]
            pct = norm_mat[i, j] * 100.0
            color = "#ffffff" if display_mat[i, j] > thresh else "#020617"

            if n_classes <= 12:
                # Show both count and percentage for <= 12 classes
                if normalize:
                    text_str = f"{pct:.1f}%\n({int(val)})"
                else:
                    text_str = f"{int(val)}\n({pct:.0f}%)" if val > 0 else "0"
            else:
                # Show compact count or % for larger vocabularies
                text_str = f"{pct:.0f}%" if normalize else (f"{int(val)}" if val > 0 else "")

            ax.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                color=color,
                fontsize=max(7, min(10, 13 - int(n_classes * 0.35))),
                fontweight="bold" if i == j else "normal",
            )

    total_diag = int(np.trace(mat))
    total_all = int(np.sum(mat))
    overall_acc = (total_diag / total_all * 100.0) if total_all > 0 else 0.0

    ax.set_title(
        f"{title}\nĐộ chính xác tổng thể (Top-1 Accuracy): {overall_acc:.2f}% ({total_diag}/{total_all})",
        color="#38bdf8",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Nhãn thực tế (True / Ground Truth)", color="#94a3b8", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlabel("Nhãn dự đoán (Predicted)", color="#94a3b8", fontsize=11, fontweight="bold", labelpad=8)

    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.2)

    plt.tight_layout()

    # Save to disk if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")

    # Encode to base64 buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64_data = base64.b64encode(buf.read()).decode("ascii")
    data_url = f"data:image/png;base64,{b64_data}"

    return {
        "image_base64": data_url,
        "image_path": output_path,
        "confusion_matrix": mat.astype(int).tolist(),
        "normalized_matrix": np.round(norm_mat, 4).tolist(),
        "class_names": list(class_names),
        "overall_accuracy": overall_acc,
        "total_samples": total_all,
        "correct_samples": total_diag,
    }
