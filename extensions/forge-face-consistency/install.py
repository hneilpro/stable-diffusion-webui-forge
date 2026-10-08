"""Installer for Forge Face Consistency.

Forge runs this file when the extension is enabled. The swap engine needs
InsightFace (buffalo_l) and onnxruntime; both are installed here if pip is
available. The inswapper model itself is NOT downloaded automatically: place
inswapper_128.onnx under models/insightface/ (the ReActor convention) or set
its path in Settings -> Forge Face Consistency. InsightFace's buffalo_l /
inswapper models are released for non-commercial research use; the same
posture as ReActor and InstantID applies.

Dependency safety: Forge pins numpy and protobuf in requirements_versions.txt,
and several of its compiled packages (scikit-image, torch ecosystem) are built
against those pins. An unconstrained `pip install insightface` drags numpy to
2.x and protobuf to 7.x, which breaks scikit-image at import
("numpy.dtype size changed"). This installer therefore passes Forge's own
pins as pip constraints so the resolver can never move them; if no compatible
resolution exists, pip fails loudly here instead of breaking Forge.
"""

import os
import subprocess
import sys
import tempfile

REQUIREMENTS = ["insightface>=0.7.3", "onnxruntime>=1.16"]
PINNED_BY_FORGE = ("numpy", "protobuf")


def _installed(module_name):
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _forge_constraints():
    """Return pip constraint lines taken from Forge's requirements_versions.txt.

    Looks for the WebUI root two levels above this extension folder and falls
    back to a numpy<2 guard when the file cannot be found.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    webui_root = os.path.dirname(os.path.dirname(here))
    versions = os.path.join(webui_root, "requirements_versions.txt")
    lines = []
    try:
        with open(versions, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                name = line.split("==")[0].split(">=")[0].split("<")[0].strip().lower()
                if name in PINNED_BY_FORGE and "==" in line:
                    lines.append(line)
    except OSError:
        pass
    if not any(line.lower().startswith("numpy") for line in lines):
        lines.append("numpy<2")
    return lines


def _install(missing):
    constraints = _forge_constraints()
    fd, path = tempfile.mkstemp(prefix="forge-face-consistency-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(constraints) + "\n")
        print(f"[FaceConsistency] installing: {', '.join(missing)} "
              f"(constraints: {', '.join(constraints)})")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing, "-c", path])
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    missing = [pkg for pkg, mod in
               (("insightface>=0.7.3", "insightface"),
                ("onnxruntime>=1.16", "onnxruntime")) if not _installed(mod)]
    if not missing:
        print("[FaceConsistency] insightface and onnxruntime already available")
        return
    try:
        _install(missing)
    except Exception as exc:
        print(f"[FaceConsistency] pip install failed ({exc}); Forge pins were "
              f"left untouched. Install manually with: pip install "
              f"{' '.join(REQUIREMENTS)} \"numpy<2\"")


if __name__ == "__main__":
    main()
else:
    main()
