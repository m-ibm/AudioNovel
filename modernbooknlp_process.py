#!/usr/bin/env python3
"""Run the official ModernBookNLP quotation-attribution pipeline on a book.

The NEWCHAPTERABC marker used by VoxNovel is replaced with equal-length spaces
before tokenization. This prevents it being treated as a lexical token while
preserving every later UTF-8 byte offset in the original source file.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import tempfile

MARKER = "NEWCHAPTERABC"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--book-id", default="Book")
    p.add_argument("--pipeline", default="entity,quote,supersense,event,coref")
    p.add_argument("--model", choices=["big", "small"], default="big")
    p.add_argument("--model-path", type=Path, default=Path.home() / "booknlp_models")
    p.add_argument("--model-params-json", default="{}", help="Additional ModernBookNLP constructor parameters as JSON")
    a = p.parse_args()
    source = a.input.resolve()
    if not source.is_file(): raise FileNotFoundError(source)
    raw_bytes = source.read_bytes()
    raw = raw_bytes.decode("utf-8")
    if not raw.strip(): raise ValueError(f"Input book is empty: {source}")
    # Replace chapter marker with same-length ASCII spaces to keep byte offsets invariant.
    # Write bytes (not text mode) so CRLF/LF line endings are preserved exactly.
    cleaned = raw.replace(MARKER, " " * len(MARKER)).encode("utf-8")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "examples").mkdir(parents=True, exist_ok=True)
    try:
        params = json.loads(a.model_params_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --model-params-json: {exc}") from exc
    if not isinstance(params, dict): raise ValueError("--model-params-json must encode an object")
    params.update({"pipeline": a.pipeline, "model": a.model, "model_path": str(a.model_path), "modern_qa": True, "direct_qa": False})
    # Do not ask ModernBookNLP to run its older direct quotation attribution.
    from booknlp.english.english_booknlp import EnglishBookNLP
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", prefix="modernbooknlp_", delete=False) as f:
        temp_input = Path(f.name)
        f.write(cleaned)
    try:
        print(f"ModernBookNLP input: {source}", flush=True)
        print(f"Output: {a.output}; book id: {a.book_id}", flush=True)
        print(f"Model parameters: {json.dumps(params, ensure_ascii=False)}", flush=True)
        processor = EnglishBookNLP(params)
        processor.process(str(temp_input), str(a.output), a.book_id)
        expected = [a.output / f"{a.book_id}.{suffix}" for suffix in ("tokens", "entities", "quotes", "book")]
        missing = [str(x) for x in expected if not x.is_file()]
        if missing: raise RuntimeError("ModernBookNLP did not produce expected outputs: " + ", ".join(missing))
        print("ModernBookNLP completed; generated .tokens, .entities, .quotes and .book.", flush=True)
    finally:
        temp_input.unlink(missing_ok=True)
    return 0

if __name__ == "__main__": raise SystemExit(main())
