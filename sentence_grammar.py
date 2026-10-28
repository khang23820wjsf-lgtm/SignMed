#!/usr/bin/env python3
"""
===============================================================================
VIETNAMESE SIGN-LANGUAGE SENTENCE GRAMMAR, N-GRAM LANGUAGE MODEL & GLOSS->TEXT
===============================================================================
Không có dataset câu VSL thật? => Tự "biến tấu" câu từ dữ liệu TỪ ĐƠN:

1. `SentenceGrammar`  — Ngữ pháp mẫu (template) theo từ loại tiếng Việt
   (CHỦ_NGỮ, THỜI_GIAN, TRIỆU_CHỨNG, ĐỘNG_TỪ, ĐỊA_ĐIỂM, ...). Từ danh sách từ đơn
   đang có, hệ thống tự sinh ra hàng nghìn câu hợp lý: "TÔI ĐAU ĐẦU",
   "BÂY_GIỜ TÔI SỐT", "TÔI CẦN BÁC_SĨ", "TÔI ĐI BỆNH_VIỆN", ...
   => Dùng để ghép các clip `.npy` từ đơn thành câu liên tục có trật tự thật.

2. `NGramLanguageModel` — Mô hình ngôn ngữ bigram (nội suy với unigram) học từ
   kho câu sinh bởi ngữ pháp. Dùng trong CTC Beam Search (shallow fusion) và
   bộ giải mã "Ghép từ" (Word-Spotting) để ưu tiên các chuỗi từ hợp ngữ pháp.

3. `glosses_to_vietnamese` — Chuyển chuỗi gloss ký hiệu (trật tự VSL) thành câu
   tiếng Việt tự nhiên: ["TÔI", "SỐT"] -> "Tôi bị sốt."
===============================================================================
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from decoder import normalize_single_gloss

BOS = "<s>"
EOS = "</s>"

# =============================================================================
# 1. LEXICON: từ loại cho các gloss trong bộ VSL400 (472 từ)
# =============================================================================

_RAW_LEXICON: Dict[str, List[str]] = {
    # Người / đại từ có thể làm chủ ngữ
    "SUBJ": [
        "TÔI", "ANH", "CHỊ", "EM", "CHÚNG_TA", "HỌ", "MẸ", "BỐ", "BÁC", "CHÚ", "CÔ",
        "DÌ", "CẬU", "BÀ_NỘI", "BÀ_NGOẠI", "ÔNG_NỘI", "ÔNG_NGOẠI", "CON_TRAI",
        "CON_GÁI", "VỢ", "CHỒNG", "CHÁU", "BẠN_THÂN", "BÁC_SĨ", "Y_TÁ", "BỆNH_NHÂN",
        "HỌC_SINH", "SINH_VIÊN",
    ],
    # Người có thể làm tân ngữ (gọi, thăm, cần, cảm ơn ...)
    "PERSON": [
        "BÁC_SĨ", "Y_TÁ", "MẸ", "BỐ", "ANH", "CHỊ", "EM", "BÁC", "CHÚ", "CÔ", "DÌ",
        "BÀ_NỘI", "BÀ_NGOẠI", "ÔNG_NỘI", "ÔNG_NGOẠI", "VỢ", "CHỒNG", "BẠN_THÂN",
        "BỆNH_NHÂN", "TÔI",
    ],
    "TIME": [
        "BÂY_GIỜ", "HÔM_NAY", "BUỔI_SÁNG", "BUỔI_TRƯA", "BUỔI_CHIỀU", "BUỔI_TỐI",
        "BAN_NGÀY", "BAN_ĐÊM", "CHIỀU", "TỐI", "TRƯA", "THỨ_HAI", "THỨ_BA", "THỨ_TƯ",
        "THỨ_NĂM", "THỨ_SÁU", "THỨ_BẢY", "CHỦ_NHẬT",
    ],
    # Triệu chứng bệnh => "bị ..."
    "SICK": ["SỐT", "HO", "NÔN_ÓI", "ỐM", "LÂY_BỆNH"],
    # Cảm giác / trạng thái => "thấy ..."
    "FEEL": [
        "MỆT", "LẠNH", "NÓNG", "YẾU", "KHỎE", "LO_LẮNG", "BĂN_KHOĂN", "XÚC_ĐỘNG",
        "ẤM", "MÁT_MẺ", "BẬN",
    ],
    "PAIN": ["ĐAU"],
    "BODY": ["ĐẦU", "CHÂN", "BÀN_TAY", "NGÓN_TAY", "CƠ_THỂ"],
    "MOTION": ["ĐI"],
    "PLACE": [
        "BỆNH_VIỆN", "NHÀ", "CHỢ", "SIÊU_THỊ", "TRƯỜNG_HỌC", "CÔNG_VIÊN", "NHÀ_HÀNG",
        "NHÀ_SÁCH", "NGÂN_HÀNG", "CÔNG_TY", "QUÁN_CÀ_PHÊ", "RẠP_CHIẾU_PHIM",
        "THÀNH_PHỐ", "TRƯỜNG_ĐẠI_HỌC", "TRƯỜNG_CAO_ĐẲNG", "KHU_CÁCH_LY", "NHÀ_TRỌ",
    ],
    "EAT": ["ĂN", "NẤU", "THÈM", "MUA", "NẾM"],
    "FOOD": [
        "PHỞ", "BÚN", "XÔI", "BÁNH_BAO", "BÁNH_XÈO", "BÁNH_CHƯNG", "BÁNH_TÉT",
        "MÌ_GÓI", "THỊT", "CÁ", "TRỨNG", "RAU", "GẠO", "KẸO", "KEM", "THỨC_ĂN",
        "QUẢ_CAM", "QUẢ_CHUỐI", "QUẢ_XOÀI", "QUẢ_DỪA", "QUẢ_BƠ",
    ],
    "DRINK": ["UỐNG"],
    "BEVERAGE": ["NƯỚC", "CÀ_PHÊ", "TRÀ", "SỮA", "BIA", "RƯỢU"],
    "NEED": ["CẦN", "MUỐN", "KHÔNG_CẦN"],
    "NEED_OBJ": [
        "BÁC_SĨ", "Y_TÁ", "NƯỚC", "GIÚP_ĐỠ", "KHẨU_TRANG", "NGHỈ_NGƠI", "KHÁM_BỆNH",
        "ĐIỆN_THOẠI", "TAXI",
    ],
    "MODAL": ["CÓ_THỂ", "NÊN", "KHÔNG_NÊN", "MUỐN", "CẦN"],
    "INTRANS": [
        "NGỦ", "CƯỜI", "KHÓC", "HÁT", "MÚA", "CHẠY", "NGHỈ_NGƠI", "LÀM_VIỆC",
        "TẬP_LUYỆN", "HỌC", "THỨC_DẬY", "TẮM_RỬA", "BƠI_LỘI", "LÀM_BÀI_TẬP",
        "XUẤT_VIỆN", "KHÁM_BỆNH",
    ],
    "TRANS_PERSON": ["GỌI", "THĂM", "GIÚP", "GIÚP_ĐỠ", "YÊU_THƯƠNG", "NHỚ", "CHỜ"],
    "SOCIAL": ["CẢM_ƠN", "XIN_LỖI", "CHÀO"],
    "KNOW": ["BIẾT", "HIỂU", "ĐỒNG_Ý", "NHỚ"],
    "READ": ["ĐỌC", "XEM", "VIẾT", "THÍCH"],
    "READ_OBJ": ["SÁCH", "TIVI", "VỞ", "ĐIỆN_THOẠI", "LAPTOP"],
    "ANSWER": ["CÓ", "ĐÚNG", "SAI", "ĐỒNG_Ý", "CẢM_ƠN", "XIN_LỖI"],
    "WHERE": ["ĐÂU"],
}

LEXICON: Dict[str, List[str]] = {
    cat: [normalize_single_gloss(w) for w in words] for cat, words in _RAW_LEXICON.items()
}

# Template = (danh sách slot, trọng số). Slot là tên từ loại (viết HOA trong LEXICON)
# hoặc một gloss cụ thể bắt đầu bằng "=" (ví dụ "=TÔI").
TEMPLATES: List[Tuple[List[str], float]] = [
    # --- Y tế / sức khoẻ (trọng tâm VSL400 medical) ---
    (["SUBJ", "SICK"], 3.0),
    (["SUBJ", "FEEL"], 2.5),
    (["SUBJ", "PAIN"], 2.8),
    (["SUBJ", "PAIN", "BODY"], 3.0),
    (["PAIN", "BODY"], 1.0),
    (["SUBJ", "PAIN", "PLACE"], 2.2),
    (["SUBJ", "SICK", "PLACE"], 1.8),
    (["SUBJ", "FEEL", "PLACE"], 1.4),
    (["SUBJ", "PLACE"], 1.8),
    (["PLACE", "SUBJ", "PAIN"], 1.2),
    (["PLACE", "SUBJ", "SICK"], 1.2),
    (["TIME", "SUBJ", "SICK"], 2.0),
    (["TIME", "SUBJ", "FEEL"], 1.6),
    (["TIME", "SUBJ", "PAIN", "BODY"], 1.6),
    (["SUBJ", "SICK", "SICK"], 1.2),
    (["SUBJ", "SICK", "FEEL"], 1.0),
    (["SUBJ", "PAIN", "BODY", "SICK"], 0.8),
    (["SUBJ", "SICK", "PAIN", "BODY"], 0.8),
    (["SUBJ", "FEEL", "SICK"], 0.8),
    (["SUBJ", "SICK", "NEED", "NEED_OBJ"], 1.0),
    (["SUBJ", "PAIN", "BODY", "NEED", "NEED_OBJ"], 0.6),
    (["SUBJ", "SICK", "MOTION", "PLACE"], 0.8),
    (["PERSON", "KNOW"], 0.6),
    (["SUBJ", "KNOW"], 0.8),
    # --- Di chuyển / nhu cầu ---
    (["SUBJ", "MOTION", "PLACE"], 2.0),
    (["TIME", "SUBJ", "MOTION", "PLACE"], 1.4),
    (["SUBJ", "NEED", "MOTION", "PLACE"], 1.0),
    (["SUBJ", "NEED", "NEED_OBJ"], 1.8),
    (["SUBJ", "MODAL", "INTRANS"], 1.0),
    (["PLACE", "WHERE"], 0.8),
    # --- Sinh hoạt hằng ngày ---
    (["SUBJ", "EAT", "FOOD"], 1.4),
    (["SUBJ", "DRINK", "BEVERAGE"], 1.4),
    (["TIME", "SUBJ", "EAT", "FOOD"], 0.8),
    (["SUBJ", "NEED", "DRINK", "BEVERAGE"], 0.6),
    (["SUBJ", "INTRANS"], 1.4),
    (["TIME", "SUBJ", "INTRANS"], 1.0),
    (["SUBJ", "READ", "READ_OBJ"], 0.8),
    (["SUBJ", "TRANS_PERSON", "PERSON"], 1.0),
    (["SOCIAL", "PERSON"], 1.0),
    (["SUBJ", "SOCIAL", "PERSON"], 0.6),
]

# Hình thức hiển thị tiếng Việt đặc biệt cho một số gloss
_SURFACE_OVERRIDES: Dict[str, str] = {
    "NÔN_ÓI": "nôn ói",
    "CAO_(NGƯỜI)": "cao",
    "CAO_(ĐỒ_VẬT)": "cao",
    "XẤU_(NGƯỜI)": "xấu",
    "XẤU_(VẬT)": "xấu",
    "ĐẸP_(NGƯỜI)": "đẹp",
    "ĐẸP_(VẬT)": "đẹp",
    "THẤP_(ĐỒ_VẬT)": "thấp",
    "HAY_(KHEN)": "hay",
    "QUẠT_(ĐỨNG)": "quạt",
    "GỐI_(ĐẦU)": "gối",
    "THỂ_DỤC_(THỂ_THAO)": "thể dục",
    "LÂY_BỆNH": "lây bệnh",
}


def gloss_surface(gloss: str) -> str:
    g = normalize_single_gloss(gloss)
    if g in _SURFACE_OVERRIDES:
        return _SURFACE_OVERRIDES[g]
    s = g.replace("_", " ").lower()
    if "(" in s:
        s = s.split("(")[0].strip()
    return s


def categories_of(gloss: str) -> List[str]:
    g = normalize_single_gloss(gloss)
    return [cat for cat, words in LEXICON.items() if g in words]


# =============================================================================
# 2. SENTENCE GRAMMAR (template-based synthesis)
# =============================================================================

class SentenceGrammar:
    """
    Sinh câu gloss hợp ngữ pháp từ một tập từ vựng bất kỳ (các lớp từ đơn đã có).
    Chỉ giữ các template mà mọi slot đều có ít nhất 1 từ thuộc từ vựng hiện có.
    """

    def __init__(self, vocabulary: Sequence[str], templates: Optional[Sequence[Tuple[List[str], float]]] = None) -> None:
        self.vocabulary: List[str] = sorted({normalize_single_gloss(w) for w in vocabulary if normalize_single_gloss(w)})
        vocab_set = set(self.vocabulary)
        self.slot_words: Dict[str, List[str]] = {
            cat: [w for w in words if w in vocab_set] for cat, words in LEXICON.items()
        }
        self.templates: List[Tuple[List[str], float]] = []
        for slots, weight in (templates or TEMPLATES):
            if all(self._slot_candidates(s) for s in slots):
                self.templates.append((list(slots), float(weight)))

    def _slot_candidates(self, slot: str) -> List[str]:
        if slot.startswith("="):
            g = normalize_single_gloss(slot[1:])
            return [g] if g in self.vocabulary else []
        return self.slot_words.get(slot, [])

    @property
    def is_usable(self) -> bool:
        return len(self.templates) > 0

    def realizable_templates(self) -> List[str]:
        return [" + ".join(slots) for slots, _ in self.templates]

    def sample(self, rng: random.Random) -> List[str]:
        if not self.templates:
            raise ValueError("Không có template ngữ pháp nào khớp với từ vựng hiện tại.")
        weights = [w for _, w in self.templates]
        slots = rng.choices([s for s, _ in self.templates], weights=weights, k=1)[0]
        out: List[str] = []
        for slot in slots:
            cands = self._slot_candidates(slot)
            # tránh lặp từ liền kề nếu còn lựa chọn khác (VD: "SỐT SỐT")
            choices = [c for c in cands if not out or c != out[-1]] or cands
            out.append(rng.choice(choices))
        return out

    def sample_many(self, n: int, seed: int = 0) -> List[List[str]]:
        rng = random.Random(seed)
        return [self.sample(rng) for _ in range(n)]

    def enumerate_all(self, limit: int = 20000) -> List[List[str]]:
        """Liệt kê (tối đa `limit`) toàn bộ câu có thể sinh ra."""
        results: List[List[str]] = []
        seen = set()
        for slots, _ in self.templates:
            stack: List[List[str]] = [[]]
            for slot in slots:
                cands = self._slot_candidates(slot)
                stack = [p + [c] for p in stack for c in cands if not p or p[-1] != c]
                if len(stack) > limit:
                    stack = stack[:limit]
            for s in stack:
                key = tuple(s)
                if key not in seen:
                    seen.add(key)
                    results.append(s)
                    if len(results) >= limit:
                        return results
        return results


# =============================================================================
# 3. N-GRAM LANGUAGE MODEL (bigram nội suy unigram, có <s> và </s>)
# =============================================================================

class NGramLanguageModel:
    """
    P(w | v) = λ · C(v, w) / C(v) + (1 - λ) · P_uni(w)
    P_uni(w) = (C(w) + k) / (N + k · |V|)   (add-k để từ chưa gặp vẫn có xác suất > 0)
    """

    def __init__(self, vocabulary: Sequence[str], lam: float = 0.85, add_k: float = 0.5) -> None:
        self.vocabulary: List[str] = sorted({normalize_single_gloss(w) for w in vocabulary if normalize_single_gloss(w)})
        self.lam = float(lam)
        self.add_k = float(add_k)
        self.unigram: Dict[str, float] = defaultdict(float)
        self.bigram: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.context_count: Dict[str, float] = defaultdict(float)
        self.total_tokens: float = 0.0
        self.num_sentences: int = 0

    def fit(self, sentences: Iterable[Sequence[str]]) -> "NGramLanguageModel":
        for sent in sentences:
            toks = [normalize_single_gloss(t) for t in sent if normalize_single_gloss(t)]
            if not toks:
                continue
            self.num_sentences += 1
            seq = [BOS] + toks + [EOS]
            for a, b in zip(seq[:-1], seq[1:]):
                self.bigram[a][b] += 1.0
                self.context_count[a] += 1.0
            for t in toks + [EOS]:
                self.unigram[t] += 1.0
                self.total_tokens += 1.0
        return self

    def _p_uni(self, w: str) -> float:
        v = len(self.vocabulary) + 1  # + </s>
        return (self.unigram.get(w, 0.0) + self.add_k) / (self.total_tokens + self.add_k * v)

    def prob(self, prev: Optional[str], word: str) -> float:
        prev_key = BOS if prev is None else normalize_single_gloss(prev) if prev not in (BOS, EOS) else prev
        w = word if word in (BOS, EOS) else normalize_single_gloss(word)
        p_uni = self._p_uni(w)
        c_ctx = self.context_count.get(prev_key, 0.0)
        if c_ctx <= 0:
            return p_uni
        p_bi = self.bigram[prev_key].get(w, 0.0) / c_ctx
        return self.lam * p_bi + (1.0 - self.lam) * p_uni

    def log_prob(self, prev: Optional[str], word: str) -> float:
        return math.log(max(self.prob(prev, word), 1e-12))

    def sentence_log_prob(self, glosses: Sequence[str]) -> float:
        total = 0.0
        prev: Optional[str] = BOS
        for g in glosses:
            total += self.log_prob(prev, g)
            prev = g
        total += self.log_prob(prev, EOS)
        return total

    def perplexity(self, sentences: Sequence[Sequence[str]]) -> float:
        lp, n = 0.0, 0
        for s in sentences:
            lp += self.sentence_log_prob(s)
            n += len(s) + 1
        return math.exp(-lp / max(1, n))

    def top_next(self, prev: Optional[str], k: int = 5) -> List[Tuple[str, float]]:
        scored = [(w, self.prob(prev, w)) for w in self.vocabulary]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:k]

    # ---- serialization ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "bigram_interpolated",
            "vocabulary": self.vocabulary,
            "lam": self.lam,
            "add_k": self.add_k,
            "unigram": dict(self.unigram),
            "bigram": {a: dict(b) for a, b in self.bigram.items()},
            "context_count": dict(self.context_count),
            "total_tokens": self.total_tokens,
            "num_sentences": self.num_sentences,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NGramLanguageModel":
        lm = cls(d.get("vocabulary", []), lam=float(d.get("lam", 0.85)), add_k=float(d.get("add_k", 0.5)))
        for k, v in (d.get("unigram") or {}).items():
            lm.unigram[k] = float(v)
        for a, row in (d.get("bigram") or {}).items():
            for b, c in row.items():
                lm.bigram[a][b] = float(c)
        for k, v in (d.get("context_count") or {}).items():
            lm.context_count[k] = float(v)
        lm.total_tokens = float(d.get("total_tokens", 0.0))
        lm.num_sentences = int(d.get("num_sentences", 0))
        return lm

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "NGramLanguageModel":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def build_language_model_for_vocabulary(
    vocabulary: Sequence[str],
    num_samples: int = 30000,
    seed: int = 7,
    lam: float = 0.85,
) -> Tuple[NGramLanguageModel, SentenceGrammar]:
    """Sinh kho câu từ ngữ pháp rồi huấn luyện bigram LM cho đúng tập từ vựng này."""
    grammar = SentenceGrammar(vocabulary)
    lm = NGramLanguageModel(vocabulary, lam=lam)
    if grammar.is_usable:
        corpus = grammar.enumerate_all(limit=20000) + grammar.sample_many(num_samples, seed=seed)
        lm.fit(corpus)
    else:
        # Không có template phù hợp => LM gần như đều (chỉ unigram add-k)
        lm.lam = 0.0
    return lm, grammar


# =============================================================================
# 4. GLOSS SEQUENCE -> CÂU TIẾNG VIỆT TỰ NHIÊN (rule-based)
# =============================================================================

_FEEL_NO_PARTICLE = {"KHỎE", "BẬN", "LO_LẮNG", "BĂN_KHOĂN", "XÚC_ĐỘNG"}


def glosses_to_vietnamese(glosses: Sequence[str]) -> str:
    """
    Chuyển chuỗi gloss VSL thành câu tiếng Việt dễ đọc, ví dụ:
      [TÔI, SỐT]                 -> "Tôi bị sốt."
      [BÂY_GIỜ, TÔI, ĐAU, ĐẦU]   -> "Bây giờ tôi bị đau đầu."
      [TÔI, MỆT]                 -> "Tôi thấy mệt."
      [TÔI, HO, SỐT]             -> "Tôi bị ho và sốt."
      [TÔI, CẦN, BÁC_SĨ]         -> "Tôi cần bác sĩ."
      [BỆNH_VIỆN, ĐÂU]           -> "Bệnh viện ở đâu?"
    """
    toks = [normalize_single_gloss(g) for g in glosses if normalize_single_gloss(g)]
    if not toks:
        return ""

    sick = set(LEXICON["SICK"])
    feel = set(LEXICON["FEEL"])
    pain = set(LEXICON["PAIN"])
    body = set(LEXICON["BODY"])
    subj = set(LEXICON["SUBJ"])
    time_w = set(LEXICON["TIME"])
    place = set(LEXICON["PLACE"])
    where = set(LEXICON["WHERE"])
    need = set(LEXICON["NEED"]) | set(LEXICON["MODAL"])

    words: List[str] = []
    prev: Optional[str] = None
    prev_was_condition = False  # vừa nói 1 triệu chứng/cảm giác => nối bằng "và"
    is_question = False

    for i, g in enumerate(toks):
        surf = gloss_surface(g)
        if g in sick or (g in pain and i + 1 < len(toks) and toks[i + 1] in body) or (g in pain and prev in subj):
            if prev_was_condition:
                words.append("và")
            elif prev in subj or prev in time_w:
                words.append("bị")
            words.append(surf)
            prev_was_condition = True
        elif g in body and prev in pain:
            words.append(surf)
            prev_was_condition = True
        elif g in feel:
            if prev_was_condition:
                words.append("và")
            elif (prev is None or prev in subj or prev in time_w) and g not in _FEEL_NO_PARTICLE:
                words.append("thấy")
            words.append(surf)
            prev_was_condition = True
        elif g in where:
            if prev in place:
                words.append("ở")
            words.append(surf)
            is_question = True
            prev_was_condition = False
        elif g in place and prev_was_condition:
            words.append(",")
            words.append("đến")
            words.append(surf)
            prev_was_condition = False
        elif g in place and prev in subj:
            words.append("đến")
            words.append(surf)
            prev_was_condition = False
        else:
            if prev_was_condition and g in need:
                words.append(",")
            words.append(surf)
            prev_was_condition = False
        prev = g

    text = " ".join(words).replace(" ,", ",")
    text = text[:1].upper() + text[1:]
    return text + ("?" if is_question else ".")


__all__ = [
    "BOS",
    "EOS",
    "LEXICON",
    "TEMPLATES",
    "SentenceGrammar",
    "NGramLanguageModel",
    "build_language_model_for_vocabulary",
    "glosses_to_vietnamese",
    "gloss_surface",
    "categories_of",
]
