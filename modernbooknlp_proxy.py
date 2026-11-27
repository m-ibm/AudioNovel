"""BookNLP-compatible constructor/process shim that delegates to ModernBookNLP."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

class BookNLP:
    """Compatibility shim for VoxNovel's ``BookNLP(language, model_params)`` API."""
    def __init__(self, language: str, model_params: dict):
        if str(language).lower() not in {"en", "english"}:
            raise ValueError("This ModernBookNLP integration currently supports English books only")
        self.model_params = dict(model_params or {})
        self.model_params["modern_qa"] = True
        self.model_params["direct_qa"] = False

    def process(self, filename, outFolder, idd):
        project_root = Path.cwd().resolve()
        script = Path(__file__).with_name("modernbooknlp_process.py").resolve()
        python = os.environ.get("VOXNOVEL_MODERNBOOKNLP_PYTHON", "/opt/modernbooknlp-env/bin/python")
        if not Path(python).is_file():
            raise FileNotFoundError(
                f"ModernBookNLP Python environment not found: {python}. "
                "Run install_modernbooknlp_in_container.sh first."
            )
        cmd = [
            python, str(script), "--input", str(Path(filename).resolve()),
            "--output", str(Path(outFolder).resolve()), "--book-id", str(idd),
            "--pipeline", str(self.model_params.get("pipeline", "entity,quote,supersense,event,coref")),
            "--model", str(self.model_params.get("model", "big")),
            "--model-path", str(self.model_params.get("model_path", Path.home() / "booknlp_models")),
            "--model-params-json", json.dumps(self.model_params, ensure_ascii=False),
        ]
        subprocess.run(cmd, cwd=project_root, check=True)
