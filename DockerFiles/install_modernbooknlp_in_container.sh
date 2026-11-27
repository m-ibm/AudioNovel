#!/usr/bin/env bash
# Install ModernBookNLP in an isolated venv INSIDE an existing VoxNovel container.
# Does not rebuild the Docker image and does not modify the container's TTS venv.
set -Eeuo pipefail
CONTAINER="${1:-voxnovel-modern}"
if ! command -v docker >/dev/null 2>&1; then echo "docker CLI not found" >&2; exit 1; fi
if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "Container '$CONTAINER' was not found. Existing containers:" >&2
  docker ps -a --format '  {{.Names}}  {{.Image}}  {{.Status}}' >&2
  exit 2
fi
RUNNING="$(docker inspect -f '{{.State.Running}}' "$CONTAINER")"
if [[ "$RUNNING" != "true" ]]; then
  echo "Container '$CONTAINER' is stopped. It may have an old generator as its command." >&2
  echo "Do NOT docker start it. First run create_cached_container.sh to make a safe idle clone." >&2
  exit 4
fi

echo "Installing ModernBookNLP inside running container '$CONTAINER' (no Dockerfile rebuild)..."
docker exec -i -u 0 "$CONTAINER" bash -s <<'IN_CONTAINER'
set -Eeuo pipefail
REPO=/opt/ModernBookNLP_QA
VENV=/opt/modernbooknlp-env
MODEL_DIR=/root/booknlp_models

if ! command -v git >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git; 
  elif command -v apk >/dev/null 2>&1; then apk add --no-cache git;
  else echo "git is required but no supported package manager was found" >&2; exit 3; fi
fi
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 --branch main https://github.com/gasmichel/ModernBookNLP_QA.git "$REPO"
else
  git -C "$REPO" pull --ff-only origin main
fi

PYTHON_BIN="$(command -v python3.11 || command -v python3.10 || command -v python3.12 || command -v python3)"
PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
"$PYTHON_BIN" - "$PYTHON_VERSION" <<'PYVER'
import sys
parts = tuple(int(x) for x in sys.argv[1].split("."))
if not ((3, 9) <= parts <= (3, 12)):
    raise SystemExit(f"ModernBookNLP dependency set requires Python 3.9-3.12; found {sys.argv[1]}. Install a compatible Python in this container, then rerun.")
PYVER
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip "setuptools<81" wheel
# Install the exact CPU PyTorch build first: this container has no CUDA GPU,
# and the requirements file pins torch==2.6.0. This avoids pulling CUDA wheels.
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
"$VENV/bin/pip" install -r "$REPO/requirements.txt"
"$VENV/bin/pip" install -e "$REPO/ModernBookNLP"
"$VENV/bin/python" -m spacy download en_core_web_sm

# The official code unconditionally enters CUDA autocast in graph attribution.
# Disable that CUDA-only context when the container has no CUDA device.
MODERN_QA="$REPO/ModernBookNLP/booknlp/english/modern_qa.py"
"$VENV/bin/python" - "$MODERN_QA" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text(encoding='utf-8')
old="with torch.autocast(device_type='cuda', dtype=DTYPE):"
new='with (torch.autocast(device_type="cuda", dtype=DTYPE) if self.device == "cuda" else __import__("contextlib").nullcontext()):'
if new not in s:
    if s.count(old) != 1:
        raise SystemExit(f"Expected exactly one CUDA autocast statement; found {s.count(old)}. Refusing an unverified patch.")
    s=s.replace(old,new)
    p.write_text(s, encoding='utf-8')
    print("Applied CPU-safe autocast guard to ModernBookNLP graph attribution.")
else:
    print("CPU-safe autocast guard already present.")
PY

mkdir -p "$MODEL_DIR" "$REPO/examples"
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
# Instantiate once to download/cache legacy entity/coreference models, ModernBERT,
# and the trained quotation-attribution checkpoint before a book is processed.
"$VENV/bin/python" - <<'PY'
from pathlib import Path
from booknlp.english.english_booknlp import EnglishBookNLP
params = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "big",
    "model_path": str(Path.home() / "booknlp_models"),
    "modern_qa": True,
    "direct_qa": False,
}
print("Downloading/initializing ModernBookNLP models; this is a one-time setup.", flush=True)
EnglishBookNLP(params)
print("ModernBookNLP model initialization succeeded.", flush=True)
PY

cat > /opt/modernbooknlp-env/INSTALL_INFO.txt <<EOF
Installed from https://github.com/gasmichel/ModernBookNLP_QA.git (main)
Python: $VENV/bin/python
Model cache: $MODEL_DIR
Modern QA: enabled (joint T=2000 model)
CPU-safe autocast patch: applied to modern_qa.py
EOF

echo "ModernBookNLP installation and model prefetch succeeded."
echo "Python: $VENV/bin/python"
echo "Models: $MODEL_DIR"
IN_CONTAINER

echo "Done. Target container ${CONTAINER} was modified; no Dockerfile/image rebuild was performed."
