"""
Reproducible tokenizer fertility and orthography audit (SLAI-029).

Evaluates a tokenizer on a native-reviewed Krio corpus without ever downloading
model weights. Computes:
  - tokens_per_word (fertility)
  - tokens_per_character
  - unknown_token_rate
  - top_fragmented_words (fragmentation of the most frequent words)
  - open_vowel_handling (tokenisation of standard SL Krio ɛ / ɔ vowels)
  - english_sequence_length_overhead (Krio vs English token ratio on parallel rows)

Refuses to run on `example_*` files, `SCHEMA_TEMPLATE` / `BENCHMARK_SCHEMA_TEMPLATE`
records, or corpora with fewer than `min_sentences` (default 1,000) native-reviewed sentences.
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

_TEMPLATE_PREFIXES = (
    "SCHEMA_TEMPLATE",
    "BENCHMARK_SCHEMA_TEMPLATE",
    "EXAMPLE_PLACEHOLDER",
)


class TokenizerLike(Protocol):
    """Minimal protocol satisfied by HuggingFace PreTrainedTokenizerBase and test doubles."""

    unk_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def tokenize(self, text: str) -> list[str]: ...


@dataclass
class TokenizerAuditReport:
    """Structured report returned by `audit_tokenizer_on_corpus` (SLAI-029)."""

    tokenizer_id: str
    corpus_path: str
    num_sentences: int
    total_words: int
    total_chars: int
    total_tokens: int
    tokens_per_word: float
    tokens_per_character: float
    unknown_token_rate: float
    top_fragmented_words: dict[str, int] = field(default_factory=dict)
    open_vowel_handling: dict[str, list[str]] = field(default_factory=dict)
    english_sequence_length_overhead: float | None = None
    note: str = (
        "Tokenizer fertility is one selection criterion among several "
        "(alongside held-out Krio benchmark quality, VRAM footprint, and licence). "
        "Earlier '≤ 2.0 tokens/word' figures are an unvalidated heuristic, not a hard pass/fail gate."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_hf_tokenizer_only(tokenizer_id: str) -> TokenizerLike:
    """
    Load ONLY tokenizer files via `transformers.AutoTokenizer.from_pretrained`.
    Never imports or calls `AutoModel*`, ensuring zero model weights are downloaded.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Tokenizer audit requires `transformers` (install via `pip install -e '.[training]'`)."
        ) from exc

    return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)


def audit_tokenizer_on_corpus(
    corpus_path: Path,
    tokenizer: TokenizerLike | None = None,
    tokenizer_id: str = "custom",
    *,
    min_sentences: int = 1000,
    require_validated: bool = True,
) -> TokenizerAuditReport:
    """
    Run the reproducible tokenizer audit on a JSONL corpus (SLAI-029).

    Raises ValueError if:
      - `corpus_path` does not exist or its filename starts with `example_`
      - Any record in the file has a `SCHEMA_TEMPLATE` / `BENCHMARK_SCHEMA_TEMPLATE` source
      - Fewer than `min_sentences` qualifying sentences are present
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    if corpus_path.stem.startswith("example_"):
        raise ValueError(
            f"Refusing to run tokenizer audit on template/example file: {corpus_path.name}"
        )

    krio_sentences: list[str] = []
    parallel_pairs: list[tuple[str, str]] = []

    with corpus_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = json.loads(stripped)
            meta = row.get("metadata", {})
            source = str(meta.get("source") or row.get("source") or "")
            if source.upper().startswith(_TEMPLATE_PREFIXES):
                raise ValueError(
                    f"Refusing to run tokenizer audit: template record found at {corpus_path.name}:{lineno}"
                )
            status = str(meta.get("validation_status") or row.get("validation_status") or "")
            if status == "rejected":
                continue
            if require_validated and status not in ("validated", "partially_validated"):
                continue

            krio_text = row.get("text") or row.get("target_text") or row.get("normalised_text")
            eng_text = row.get("source_text") or row.get("english_translation")
            if isinstance(krio_text, str) and krio_text.strip():
                norm_krio = unicodedata.normalize("NFC", krio_text.strip())
                krio_sentences.append(norm_krio)
                if isinstance(eng_text, str) and eng_text.strip():
                    parallel_pairs.append((norm_krio, unicodedata.normalize("NFC", eng_text.strip())))

    if len(krio_sentences) < min_sentences:
        raise ValueError(
            f"Corpus '{corpus_path.name}' has only {len(krio_sentences)} qualifying native-reviewed "
            f"sentence(s); tokenizer audit requires at least {min_sentences} sentences."
        )

    tok = tokenizer if tokenizer is not None else load_hf_tokenizer_only(tokenizer_id)
    unk_id = getattr(tok, "unk_token_id", None)

    total_words = 0
    total_chars = 0
    total_tokens = 0
    unk_tokens = 0
    word_counter: Counter[str] = Counter()

    for sent in krio_sentences:
        words = sent.split()
        total_words += len(words)
        total_chars += len(sent)
        for w in words:
            word_counter[w.lower()] += 1
        token_ids = tok.encode(sent, add_special_tokens=False)
        total_tokens += len(token_ids)
        if unk_id is not None:
            unk_tokens += sum(1 for tid in token_ids if tid == unk_id)

    # Fragmentation of the 20 most frequent words
    top_fragmented: dict[str, int] = {}
    for word, _ in word_counter.most_common(20):
        subtokens = tok.tokenize(word)
        top_fragmented[word] = len(subtokens)

    # Open vowel (ɛ / ɔ) handling probe on canonical single-character & syllable strings
    open_vowel_handling = {
        "ɛ": tok.tokenize("ɛ"),
        "ɔ": tok.tokenize("ɔ"),
        "Ɛ": tok.tokenize("Ɛ"),
        "Ɔ": tok.tokenize("Ɔ"),
    }

    # Sequence-length overhead versus English on parallel sentences
    overhead: float | None = None
    if parallel_pairs:
        krio_tok_sum = sum(len(tok.encode(k, add_special_tokens=False)) for k, _ in parallel_pairs)
        eng_tok_sum = sum(len(tok.encode(e, add_special_tokens=False)) for _, e in parallel_pairs)
        if eng_tok_sum > 0:
            overhead = round(krio_tok_sum / eng_tok_sum, 4)

    return TokenizerAuditReport(
        tokenizer_id=tokenizer_id,
        corpus_path=str(corpus_path),
        num_sentences=len(krio_sentences),
        total_words=total_words,
        total_chars=total_chars,
        total_tokens=total_tokens,
        tokens_per_word=round(total_tokens / max(total_words, 1), 4),
        tokens_per_character=round(total_tokens / max(total_chars, 1), 4),
        unknown_token_rate=round(unk_tokens / max(total_tokens, 1), 6),
        top_fragmented_words=top_fragmented,
        open_vowel_handling=open_vowel_handling,
        english_sequence_length_overhead=overhead,
    )
