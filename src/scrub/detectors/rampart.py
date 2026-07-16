"""Rampart ML detector — pure-Python (ONNX + tokenizers) port of the
Rampart token-classification pipeline.

The model (`nationaldesignstudio/rampart`) is a small uncased BERT that emits
BIO token labels for 17 PII entity types. We:

  1. tokenize the whole text once (char offsets come from the tokenizer),
  2. chunk the token stream into overlapping windows <= 512 positions,
  3. run batched ONNX inference, resolving overlap regions in favour of the
     chunk where a token sits furthest from a chunk boundary,
  4. softmax + BIO-decode into spans over ABSOLUTE character offsets,
  5. repair spans the way Rampart's upstream post-processing does
     (adjacent merge, single-token bridge merge, word-boundary expansion).

No hardcoded label list: `id2label` is read from the repo's config.json.
The model is loaded lazily on the first detect() or via warmup() (the daemon
calls warmup() so the first real request is fast).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from ..config import RAMPART_REPO, RAMPART_REVISION, Config
from ..types import EntityType, Span

# Model geometry (validated in ARCHITECTURE.md).
_MAX_POSITIONS = 512
_SPECIAL_SLOTS = 2  # [CLS] ... [SEP]
_CONTENT_WINDOW = 448  # content tokens per chunk (<= 510)
_STRIDE = 384  # step between chunk starts => 64-token overlap
_OVERLAP = _CONTENT_WINDOW - _STRIDE  # 64

# Characters that may sit between two same-type spans and still be treated as
# a single entity during the "adjacent merge" repair step.
_JOINERS = frozenset(" \t\n\r-'’")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis, numerically stable."""
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


class RampartDetector:
    """Detector protocol implementation (name="rampart")."""

    name = "rampart"

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        self._tokenizer: Tokenizer | None = None
        self._session: ort.InferenceSession | None = None
        # id -> label string (e.g. "B-GIVEN_NAME"); loaded from config.json.
        self._id2label: dict[int, str] = {}
        # id -> EntityType (or None for "O"); derived from _id2label.
        self._id2type: dict[int, EntityType | None] = {}
        self._num_labels = 0
        self._cls_id = 2
        self._sep_id = 3
        self._pad_id = 0

    # ------------------------------------------------------------------ init

    def warmup(self) -> None:
        """Load tokenizer, ONNX session and label map. Idempotent."""
        if self._session is not None:
            return
        model_dir = Path(snapshot_download(RAMPART_REPO, revision=RAMPART_REVISION))

        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # We do our own chunking; make sure the tokenizer never truncates/pads.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        for name, attr in (("[CLS]", "_cls_id"), ("[SEP]", "_sep_id"), ("[PAD]", "_pad_id")):
            tid = self._tokenizer.token_to_id(name)
            if tid is not None:
                setattr(self, attr, tid)

        cfg = json.loads((model_dir / "config.json").read_text())
        id2label = cfg["id2label"]
        self._id2label = {int(k): v for k, v in id2label.items()}
        self._num_labels = len(self._id2label)
        self._id2type = {}
        for i, label in self._id2label.items():
            if label == "O" or "-" not in label:
                self._id2type[i] = None
                continue
            _, raw_type = label.split("-", 1)
            self._id2type[i] = EntityType(raw_type)  # names match EntityType exactly

        self._session = ort.InferenceSession(
            str(model_dir / "onnx" / "model_q4.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def _ensure_ready(self) -> None:
        if self._session is None:
            self.warmup()

    # --------------------------------------------------------------- detect

    def detect(self, text: str) -> list[Span]:
        self._ensure_ready()
        assert self._tokenizer is not None and self._session is not None

        if not text.strip():
            return []

        enc = self._tokenizer.encode(text)
        # Keep only real content tokens (drop [CLS]/[SEP] and any (0,0) offset).
        content_ids: list[int] = []
        content_offsets: list[tuple[int, int]] = []
        for tid, (a, b), special in zip(enc.ids, enc.offsets, enc.special_tokens_mask):
            if special or (a == 0 and b == 0):
                continue
            content_ids.append(tid)
            content_offsets.append((a, b))

        n = len(content_ids)
        if n == 0:
            return []

        probs = self._infer_probs(content_ids)  # (n, num_labels), overlap-resolved
        spans = self._decode(probs, content_offsets, text)
        return spans

    # ------------------------------------------------------------ inference

    def _infer_probs(self, content_ids: list[int]) -> np.ndarray:
        """Run chunked, batched inference. Returns (n_tokens, num_labels)
        softmax probabilities, one row per content token, choosing — in overlap
        regions — the chunk where the token is furthest from a chunk boundary."""
        assert self._session is not None
        n = len(content_ids)

        # Build chunk windows over global content-token indices.
        chunks: list[tuple[int, int]] = []  # (start, end) exclusive
        start = 0
        while start < n:
            end = min(start + _CONTENT_WINDOW, n)
            chunks.append((start, end))
            if end == n:
                break
            start += _STRIDE

        # Assemble padded batch: [CLS] + content + [SEP].
        max_len = max(e - s for s, e in chunks) + _SPECIAL_SLOTS
        batch = len(chunks)
        input_ids = np.full((batch, max_len), self._pad_id, dtype=np.int64)
        attn = np.zeros((batch, max_len), dtype=np.int64)
        type_ids = np.zeros((batch, max_len), dtype=np.int64)
        for bi, (s, e) in enumerate(chunks):
            seq = [self._cls_id, *content_ids[s:e], self._sep_id]
            input_ids[bi, : len(seq)] = seq
            attn[bi, : len(seq)] = 1

        logits = self._session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attn, "token_type_ids": type_ids},
        )[0]  # (batch, max_len, num_labels)

        probs = np.zeros((n, self._num_labels), dtype=np.float32)
        # Distance-from-boundary of the token currently owning each row; -1 = unset.
        best_dist = np.full(n, -1, dtype=np.int64)
        for bi, (s, e) in enumerate(chunks):
            length = e - s
            chunk_probs = _softmax(logits[bi, 1 : 1 + length])  # skip [CLS]; drop [SEP]/pad
            for j in range(length):
                gidx = s + j
                dist = min(j, length - 1 - j)
                if dist > best_dist[gidx]:
                    best_dist[gidx] = dist
                    probs[gidx] = chunk_probs[j]
        return probs

    # -------------------------------------------------------------- decoding

    def _decode(
        self,
        probs: np.ndarray,
        offsets: list[tuple[int, int]],
        text: str,
    ) -> list[Span]:
        argmax = probs.argmax(axis=1)
        conf = probs[np.arange(len(probs)), argmax]

        # Raw BIO decode (lenient: I- after O / different type starts a span).
        # Each raw span: [type, [token global indices]].
        raw: list[tuple[EntityType, list[int]]] = []
        cur_type: EntityType | None = None
        cur_tokens: list[int] = []

        def flush() -> None:
            nonlocal cur_type, cur_tokens
            if cur_type is not None and cur_tokens:
                raw.append((cur_type, cur_tokens))
            cur_type = None
            cur_tokens = []

        for i, label_id in enumerate(argmax):
            label = self._id2label[int(label_id)]
            etype = self._id2type[int(label_id)]
            if etype is None:  # "O"
                flush()
                continue
            is_begin = label.startswith("B-")
            if is_begin or etype != cur_type:
                flush()
                cur_type = etype
                cur_tokens = [i]
            else:  # I- continuing same type
                cur_tokens.append(i)
        flush()

        # (b) bridge-and-merge and (a) adjacent merge operate on token index
        # runs; do bridge first (needs single-token gaps), then adjacent.
        raw = self._bridge_merge(raw, argmax, probs)
        spans = self._runs_to_spans(raw, offsets, conf)
        spans = self._adjacent_merge(spans, text)
        spans = self._expand_to_words(spans, text)

        # Threshold on span confidence (mean of token confidences), with
        # per-type floors from config (stricter for types that misfire on
        # source code — see Config.rampart_type_thresholds).
        base = self._config.rampart_confidence
        overrides = self._config.rampart_type_thresholds
        spans = [
            s
            for s in spans
            if s.confidence >= max(base, overrides.get(s.entity_type, base))
        ]

        # Latin-script personal names are capitalized; a name span with no
        # uppercase letter is almost always a code/prose token misread (the
        # model is uncased and scores e.g. 'daemon' ~0.52 as GIVEN_NAME).
        # Confidence can't separate these from real names in awkward contexts
        # (which score as low as ~0.42), but capitalization does.
        spans = [
            s
            for s in spans
            if s.entity_type not in (EntityType.GIVEN_NAME, EntityType.SURNAME)
            or any(c.isupper() for c in s.text)
        ]
        spans.sort(key=lambda s: (s.start, s.end))
        return spans

    def _bridge_merge(
        self,
        raw: list[tuple[EntityType, list[int]]],
        argmax: np.ndarray,
        probs: np.ndarray,
    ) -> list[tuple[EntityType, list[int]]]:
        """(b) Merge two same-type spans separated by exactly ONE token whose
        SECOND-best label is that same type. Best-effort; bridges the gap token
        into the merged run."""
        if not raw:
            return raw
        merged: list[tuple[EntityType, list[int]]] = [raw[0]]
        for etype, tokens in raw[1:]:
            ptype, ptokens = merged[-1]
            gap_start = ptokens[-1] + 1
            gap_end = tokens[0]
            if ptype == etype and gap_end - gap_start == 1:
                gap = gap_start
                # second-best label of the gap token
                order = np.argsort(probs[gap])[::-1]
                second = int(order[1])
                if self._id2type.get(second) == etype:
                    merged[-1] = (ptype, [*ptokens, gap, *tokens])
                    continue
            merged.append((etype, tokens))
        return merged

    def _runs_to_spans(
        self,
        raw: list[tuple[EntityType, list[int]]],
        offsets: list[tuple[int, int]],
        conf: np.ndarray,
    ) -> list[Span]:
        spans: list[Span] = []
        for etype, tokens in raw:
            start = offsets[tokens[0]][0]
            end = max(offsets[t][1] for t in tokens)
            mean_conf = float(np.mean([conf[t] for t in tokens]))
            spans.append(
                Span(
                    start=start,
                    end=end,
                    entity_type=etype,
                    text="",  # filled after edge expansion
                    confidence=mean_conf,
                    source="rampart",
                )
            )
        return spans

    def _adjacent_merge(self, spans: list[Span], text: str) -> list[Span]:
        """(a) Merge same-type spans separated only by joiner characters
        (whitespace / hyphen / apostrophe)."""
        if not spans:
            return spans
        spans = sorted(spans, key=lambda s: (s.start, s.end))
        out: list[Span] = [spans[0]]
        for s in spans[1:]:
            prev = out[-1]
            gap = text[prev.end : s.start]
            if s.entity_type == prev.entity_type and prev.end <= s.start and all(
                c in _JOINERS for c in gap
            ):
                # weight confidence by span char length (proxy for token count)
                lp, ls = len(prev), len(s)
                new_conf = (prev.confidence * lp + s.confidence * ls) / max(lp + ls, 1)
                out[-1] = Span(
                    start=prev.start,
                    end=max(prev.end, s.end),
                    entity_type=prev.entity_type,
                    text="",
                    confidence=new_conf,
                    source="rampart",
                )
            else:
                out.append(s)
        return out

    def _expand_to_words(self, spans: list[Span], text: str) -> list[Span]:
        """(c) Expand span edges to word boundaries so a wordpiece boundary
        never clips a word (e.g. 'aria Garcia' -> 'Maria Garcia')."""
        n = len(text)
        out: list[Span] = []
        for s in spans:
            start, end = s.start, s.end
            while start > 0 and _is_word_char(text[start - 1]) and _is_word_char(text[start]):
                start -= 1
            while end < n and _is_word_char(text[end - 1]) and _is_word_char(text[end]):
                end += 1
            out.append(
                Span(
                    start=start,
                    end=end,
                    entity_type=s.entity_type,
                    text=text[start:end],
                    confidence=s.confidence,
                    source="rampart",
                )
            )
        return out
