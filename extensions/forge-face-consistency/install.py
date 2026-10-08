"""Installer for Forge Face Consistency.

Forge runs this file when the extension is enabled. The swap engine needs
InsightFace (buffalo_l) and onnxruntime; both are installed here if pip is
available. The inswapper model itself is NOT downloaded automatically: place
inswapper_128.onnx under models/insightface/ (the ReActor convention) or set
its path in Settings -> Forge Face Consistency. InsightFace's buffalo_l /
inswapper models are released for non-commercial research use; the same
posture as ReActor and InstantID applies.
"""

import subprocess
import sys

REQUIREMENTS = ["insightface>=0.7.3", "onnxruntime>=1.16"]


def _installed(module_name):
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def main():
    missing = [pkg for pkg, mod in
               (("insightface>=0.7.3", "insightface"),
                ("onnxruntime>=1.16", "onnxruntime")) if not _installed(mod)]
    if not missing:
        print("[FaceConsistency] insightface and onnxruntime already available")
        return
    print(f"[FaceConsistency] installing: {', '.join(missing)}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing])
    except Exception as exc:
        print(f"[FaceConsistency] pip install failed ({exc}); install manually: "
              f"pip install {' '.join(REQUIREMENTS)}")


if __name__ == "__main__":
    main()
else:
    main()
