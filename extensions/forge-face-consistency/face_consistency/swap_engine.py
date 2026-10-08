"""InsightFace swap engine, ported from the proven standalone swap.py.

Measured behaviour carried over from that engine (live on the owner's
4090 outputs): buffalo_l detection, inswapper_128 swap with paste_back,
multi-reference averaged identity template with outlier cleaning (refs
below 0.35 cosine to the group mean are dropped), ArcFace cosine verify
before/after, and mandatory restoration after the 128px swap (GFPGAN 1.4
ONNX when present, else a high-frequency detail graft inside a feathered
face mask).

InsightFace / onnxruntime / cv2 are imported lazily inside the functions
that need them, so ``import face_consistency.swap_engine`` (and the pure
helpers below) work in a bare offline test environment.
"""

from __future__ import annotations

import os

import numpy as np

OUTLIER_THRESHOLD = 0.35
SAME_PERSON_THRESHOLD = 0.55

# Detection resolution. buffalo_l's default (640) is fast; faces that are
# small relative to the frame (full-body generations) are re-tried at 1280.
DEFAULT_DET_SIZE = (640, 640)
LARGE_DET_SIZE = (1280, 1280)
LARGE_IMAGE_MIN = 768  # max(H, W) at/above this enables the 1280 retry


def _ort_providers():
    """onnxruntime providers: CUDA first when available, CPU always last.

    onnxruntime falls back down the list automatically, so a machine
    without onnxruntime-gpu simply runs on CPU — never a crash.
    """
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def _embedding_of(face_or_array):
    if isinstance(face_or_array, np.ndarray):
        return np.asarray(face_or_array, dtype=np.float64)
    emb = getattr(face_or_array, "normed_embedding", None)
    if emb is None:
        emb = getattr(face_or_array, "embedding")
    return np.asarray(emb, dtype=np.float64)


def template_embedding(faces):
    """Outlier-cleaned average (template) embedding, unit-normalised.

    ``faces`` may be InsightFace face objects or raw embedding arrays.
    Returns ``(mean_embedding, keep_indices)``. A reference that does not
    agree with the group (< 0.35 cosine to the mean) is dropped once, so
    one stray wrong photo cannot mush the identity. Pure numpy: safe to
    unit-test offline.
    """
    embs = [_embedding_of(f) for f in faces]
    if not embs:
        raise ValueError("template_embedding needs at least one embedding")
    mean = np.mean(embs, axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    keep = [i for i, e in enumerate(embs)
            if float(np.dot(mean, e)) >= OUTLIER_THRESHOLD]
    if not keep:
        keep = list(range(len(embs)))
    mean = np.mean([embs[i] for i in keep], axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean, keep


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def largest(faces):
    """Face with the largest bounding-box area."""
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def collect_ref_faces(app, source):
    """Every detected reference face, largest per file.

    ``source`` is one image path, a comma list of paths, a directory of
    images, or a list of paths. Returns a list of (path, image, face).
    Requires cv2 and a prepared FaceAnalysis ``app``.
    """
    import cv2

    if isinstance(source, (list, tuple)):
        paths = list(source)
    elif isinstance(source, str) and os.path.isdir(source):
        paths = sorted(
            os.path.join(source, f) for f in os.listdir(source)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
    elif isinstance(source, str) and "," in source:
        paths = [x.strip() for x in source.split(",") if x.strip()]
    else:
        paths = [source]
    found = []
    for path in paths:
        if not isinstance(path, str) or not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        faces = app.get(img)
        if faces:
            found.append((path, img, largest(faces)))
    return found


def face_sharpness(image, face):
    """Laplacian variance of the face crop (blur ~20-30, sharp 80+)."""
    import cv2

    if face is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in face.bbox)
    crop = image[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)


def restore_face(target_before_swap, swapped, face):
    """Graft the target's own high-frequency detail back after a swap."""
    import cv2

    x0, y0, x1, y1 = (int(v) for v in face.bbox)
    mask = np.zeros(swapped.shape[:2], np.float32)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 8)[..., None]
    blur = cv2.GaussianBlur(target_before_swap, (0, 0), 3)
    detail = target_before_swap.astype(np.float32) - blur.astype(np.float32)
    out = swapped.astype(np.float32) + detail * 0.6 * mask
    soft = cv2.GaussianBlur(out, (0, 0), 2)
    unsharp = out * 1.5 - soft * 0.5
    out = out * (1 - mask) + unsharp * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def gfpgan_restore(image, face, model_path):
    """GFPGAN 1.4 ONNX restoration; returns (image, used_model)."""
    import cv2

    if not model_path or not os.path.isfile(model_path):
        return image, False
    import onnxruntime as ort
    from insightface.utils import face_align

    aligned, M = face_align.norm_crop2(image, face.kps, image_size=512)
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = (rgb / 255.0 - 0.5) / 0.5
    blob = np.transpose(blob, (2, 0, 1))[None, ...]
    sess = ort.InferenceSession(model_path, providers=_ort_providers())
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
    out = np.clip((np.transpose(out, (1, 2, 0)) * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    restored = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    Minv = cv2.invertAffineTransform(M)
    warped = cv2.warpAffine(restored, Minv, (image.shape[1], image.shape[0]))
    mask = np.full((512, 512), 255, np.uint8)
    mask[:8, :] = mask[-8:, :] = 0
    mask[:, :8] = mask[:, -8:] = 0
    mask = cv2.GaussianBlur(mask, (0, 0), 12)
    m = cv2.warpAffine(mask, Minv, (image.shape[1], image.shape[0])).astype(np.float32)[..., None] / 255.0
    blended = (image.astype(np.float32) * (1 - m) + warped.astype(np.float32) * m).astype(np.uint8)
    return blended, True


def default_model_candidates(forge_root=None):
    """Candidate paths for inswapper / GFPGAN, ReActor convention first."""
    roots = []
    if forge_root:
        roots.append(forge_root)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots.append(os.path.dirname(here))
    roots.append(os.getcwd())
    inswapper, gfpgan = [], []
    for root in roots:
        inswapper.append(os.path.join(root, "models", "insightface", "inswapper_128.onnx"))
        gfpgan.append(os.path.join(root, "models", "insightface", "gfpgan_1.4.onnx"))
        inswapper.append(os.path.join(root, "extensions", "forge-face-consistency", "models", "inswapper_128.onnx"))
        gfpgan.append(os.path.join(root, "extensions", "forge-face-consistency", "models", "gfpgan_1.4.onnx"))
    inswapper.append(os.path.join(here, "models", "inswapper_128.onnx"))
    gfpgan.append(os.path.join(here, "models", "gfpgan_1.4.onnx"))
    return inswapper, gfpgan


def _first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


class FaceSwapEngine:
    """Lazy InsightFace engine: detection + swap + verify + restore."""

    def __init__(self, inswapper_path=None, gfpgan_path=None, forge_root=None,
                 det_size=DEFAULT_DET_SIZE):
        cands_in, cands_gf = default_model_candidates(forge_root)
        self.inswapper_path = inswapper_path or _first_existing(cands_in)
        self.gfpgan_path = gfpgan_path or _first_existing(cands_gf)
        self._app = None
        self._app_ctx = -1
        self._det_size = tuple(det_size)
        self._swapper = None
        # Template cache: building the averaged identity (detect + embed
        # every reference) is the expensive part and is identical for
        # every image in a batch. Keyed by the ref-source identity so a
        # cached template is never reused across different references.
        self._template_cache = {}

    @staticmethod
    def _template_key(ref_source):
        if isinstance(ref_source, np.ndarray):
            # Single in-memory image: key on identity + shape so a new
            # array never collides with a stale cached template.
            return ("ndarray", id(ref_source), ref_source.shape)
        return ("path", ref_source)

    @property
    def app(self):
        if self._app is None:
            from insightface.app import FaceAnalysis

            providers = _ort_providers()
            ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=self._det_size)
            self._app = app
            self._app_ctx = ctx_id
        return self._app

    def _prepare_det_size(self, det_size):
        det_size = tuple(det_size)
        if det_size != self._det_size:
            self.app.prepare(ctx_id=self._app_ctx, det_size=det_size)
            self._det_size = det_size

    def detect(self, image_bgr):
        """Detect faces, escalating resolution for large images.

        Standard pass at the default det_size; when nothing is found in
        a large frame (typical for full-body generations where the face
        is small), re-try once at 1280 before giving up. The detector is
        restored to the default size afterwards.
        """
        faces = self.app.get(image_bgr)
        if not faces:
            h, w = image_bgr.shape[:2]
            if max(h, w) >= LARGE_IMAGE_MIN and self._det_size[0] < LARGE_DET_SIZE[0]:
                prev = self._det_size
                try:
                    self._prepare_det_size(LARGE_DET_SIZE)
                    faces = self.app.get(image_bgr)
                finally:
                    self._prepare_det_size(prev)
        return faces

    @property
    def swapper(self):
        if self._swapper is None:
            from insightface.model_zoo import get_model

            if not self.inswapper_path:
                raise FileNotFoundError(
                    "inswapper_128.onnx not found; place it under "
                    "models/insightface/ (ReActor convention)")
            self._swapper = get_model(self.inswapper_path,
                                      providers=_ort_providers())
        return self._swapper

    def build_template(self, ref_source):
        """Averaged identity template from a path / dir / comma list.

        ``ref_source`` may also be an already-decoded BGR numpy image.
        Returns (template_embedding, num_sources_used, base_face).
        Results are cached per ref_source so a batch reuses one template.
        """
        key = self._template_key(ref_source)
        if key in self._template_cache:
            return self._template_cache[key]
        result = self._build_template_uncached(ref_source)
        self._template_cache[key] = result
        return result

    def _build_template_uncached(self, ref_source):
        import cv2

        app = self.app
        if isinstance(ref_source, np.ndarray):
            faces = app.get(ref_source)
            if not faces:
                raise ValueError("no face detected in reference image")
            face = largest(faces)
            return np.asarray(face.normed_embedding), 1, face
        refs = collect_ref_faces(app, ref_source)
        if not refs:
            raise ValueError("no face detected in reference(s)")
        mean_emb, keep = template_embedding([f for _, _, f in refs])
        # Drive the swapper with the sharpest kept reference face carrying
        # the averaged identity: a sharp source gives inswapper cleaner
        # geometry than the first file in sorted order.
        kept = [refs[i] for i in keep]
        base_face = max(
            kept, key=lambda t: face_sharpness(t[1], t[2]) or -1.0)[2]
        # Drive the swapper with the averaged identity, scaled like the base.
        base_face.embedding = mean_emb * float(np.linalg.norm(base_face.embedding) or 1.0)
        base_face.normed_embedding = mean_emb
        return mean_emb, len(keep), base_face

    def similarity(self, ref_embedding, image_bgr):
        """(cosine vs template, face count, largest face or None)."""
        faces = self.detect(image_bgr)
        if not faces:
            return None, 0, None
        face = largest(faces)
        return cosine(ref_embedding, face.normed_embedding), len(faces), face

    def swap(self, target_bgr, ref_source, restore=True):
        """Full swap + restore. Returns (out_bgr, info dict)."""
        template, n_sources, src_face = self.build_template(ref_source)
        sim_before, nfaces, dst = self.similarity(template, target_bgr)
        if nfaces == 0 or dst is None:
            raise ValueError("no face detected in target image")
        h, w = target_bgr.shape[:2]
        bw = max(0.0, dst.bbox[2] - dst.bbox[0])
        bh = max(0.0, dst.bbox[3] - dst.bbox[1])
        face_size_ratio = round((bw * bh) / max(1, h * w), 4)
        sharp_before = face_sharpness(target_bgr, dst)
        out = self.swapper.get(target_bgr, dst, src_face, paste_back=True)
        restored_by = "none"
        sharp_swapped = None
        if restore:
            out_faces = self.detect(out)
            if out_faces:
                out_face = largest(out_faces)
                sharp_swapped = face_sharpness(out, out_face)
                out, used = gfpgan_restore(out, out_face, self.gfpgan_path)
                if used:
                    restored_by = "gfpgan"
                else:
                    out = restore_face(target_bgr, out, out_face)
                    restored_by = "detail-graft"
        sim_after, _, _ = self.similarity(template, out)
        out_faces = self.detect(out)
        info = {
            "similarity_before": sim_before,
            "similarity_after": sim_after,
            "target_faces": nfaces,
            "identity_sources": n_sources,
            "face_size_ratio": face_size_ratio,
            "face_sharpness_before": sharp_before,
            "face_sharpness_swapped": sharp_swapped,
            "face_sharpness": face_sharpness(out, largest(out_faces)) if out_faces else None,
            "restored": bool(restore),
            "restored_by": restored_by,
        }
        return out, info
