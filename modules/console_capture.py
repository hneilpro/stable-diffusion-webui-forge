"""In-process console capture for stable-diffusion-webui-forge.

Captures everything the server prints to the console — ``print()`` output on stdout/stderr,
``logging`` records, and uncaught tracebacks — into a bounded, thread-safe ring buffer that can
be queried over the REST API (``GET /sdapi/v1/console-log``) and shown in the "Console" UI tab.

Stdlib only. Importing this module has no side effects; call :func:`install` to activate.

Bounded by construction: at most ``max_lines`` entries and ``max_bytes`` of text. Capture
handlers never raise into the application (a re-entrancy guard plus broad try/except).
"""

import collections
import io
import itertools
import logging
import sys
import threading
import time
import traceback

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 512 * 1024
MAX_LINE_CHARS = 8192
MAX_PARTIAL_CHARS = 4096


class ConsoleCapture:
    """Thread-safe ring buffer of console lines with a monotonic cursor."""

    def __init__(self, max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES):
        self._max_lines = max(1, int(max_lines))
        self._max_bytes = max(1024, int(max_bytes))
        self._entries = collections.deque()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._total_bytes = 0
        self._local = threading.local()

    # -- writing ---------------------------------------------------------
    def _guarded(self):
        """Re-entrancy guard: capture code must never capture its own output."""
        if getattr(self._local, "inside", False):
            return None
        self._local.inside = True
        return self._local

    def add(self, text, stream="stdout", level="INFO"):
        """Append one logical line. Never raises."""
        token = self._guarded()
        if token is None:
            return
        try:
            if not isinstance(text, str):
                text = str(text)
            if len(text) > MAX_LINE_CHARS:
                text = text[:MAX_LINE_CHARS] + " ...[truncated]"
            size = len(text.encode("utf-8", "replace"))
            with self._lock:
                # ID assignment happens under the lock so append order always matches
                # ID order; cursor-based pagination can never skip a line.
                entry = {
                    "id": next(self._ids),
                    "ts": time.time(),
                    "stream": stream,
                    "level": level,
                    "text": text,
                }
                self._entries.append(entry)
                self._total_bytes += size
                while len(self._entries) > self._max_lines or (
                    self._total_bytes > self._max_bytes and len(self._entries) > 1
                ):
                    old = self._entries.popleft()
                    self._total_bytes -= len(old["text"].encode("utf-8", "replace"))
        except Exception:
            pass
        finally:
            token.inside = False

    def add_exception(self, exc_type, exc_value, tb):
        """Format and store a full traceback as a single ERROR entry. Never raises."""
        try:
            text = "".join(traceback.format_exception(exc_type, exc_value, tb)).rstrip("\n")
        except Exception:
            text = f"{exc_type.__name__}: {exc_value}"
        # Also feed the structured store in modules.errors (lazy import: keep this module standalone).
        try:
            from modules import errors as _errors

            if hasattr(_errors, "record_exception_info"):
                _errors.record_exception_info(exc_type, exc_value, tb)
        except Exception:
            pass
        self.add(text, stream="stderr", level="ERROR")

    # -- reading ---------------------------------------------------------
    @property
    def cursor(self):
        """The id that a subsequent ``since`` query should use (0 when empty)."""
        with self._lock:
            return self._entries[-1]["id"] if self._entries else 0

    def get(self, since=0, limit=200):
        """Return ``(cursor, entries)`` with ``entry["id"] > since``, oldest first."""
        try:
            since = int(since)
        except (TypeError, ValueError):
            since = 0
        limit = max(1, min(int(limit), DEFAULT_MAX_LINES))
        with self._lock:
            if since <= 0:
                entries = list(self._entries)[-limit:]
            else:
                entries = [e for e in self._entries if e["id"] > since][-limit:]
            cursor = self._entries[-1]["id"] if self._entries else 0
        return cursor, [dict(e) for e in entries]

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0


class _CaptureLogHandler(logging.Handler):
    """logging.Handler that forwards formatted records into a ConsoleCapture."""

    def __init__(self, capture):
        super().__init__(level=logging.NOTSET)
        self.capture = capture
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    def emit(self, record):
        try:
            text = self.format(record)
        except Exception:
            try:
                text = record.getMessage()
            except Exception:
                return
        stream = "stderr" if record.levelno >= logging.ERROR else "stdout"
        self.capture.add(text, stream=stream, level=record.levelname)


class _TeeStream(io.TextIOBase):
    """A sys.stdout/sys.stderr proxy that tees writes into a ConsoleCapture.

    Delegates every attribute except ``write``/``writelines`` to the wrapped stream, so
    ``.buffer``, ``.fileno()``, ``.isatty()`` etc. keep working for other libraries.
    """

    def __init__(self, capture, stream, name, default_level):
        # Bypass __setattr__-less TextIOBase quirks: set via object.__setattr__.
        object.__setattr__(self, "_capture", capture)
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_default_level", default_level)
        object.__setattr__(self, "_partial", "")
        object.__setattr__(self, "_partial_lock", threading.Lock())

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_stream"), name)

    def _flush_partial_line(self, text):
        """Handle one chunk of text that may contain newlines/CRs; return leftover partial."""
        partial = object.__getattribute__(self, "_partial")
        text = partial + text
        object.__setattr__(self, "_partial", "")
        lines = text.split("\n")
        # Last element is the new partial (no trailing newline yet).
        for line in lines[:-1]:
            self._emit_line(line)
        leftover = lines[-1]
        if len(leftover) > MAX_PARTIAL_CHARS:
            # A never-terminated progress line: keep only the tail so we stay bounded.
            leftover = leftover[-MAX_PARTIAL_CHARS:]
        object.__setattr__(self, "_partial", leftover)

    def _emit_line(self, line):
        # Coalesce \r progress rewrites: keep only what follows the last carriage return.
        if "\r" in line:
            line = line.rsplit("\r", 1)[-1]
        line = line.rstrip("\r")
        if not line.strip():
            return
        capture = object.__getattribute__(self, "_capture")
        capture.add(
            line,
            stream=object.__getattribute__(self, "_name"),
            level=object.__getattribute__(self, "_default_level"),
        )

    def write(self, s):
        stream = object.__getattribute__(self, "_stream")
        if isinstance(s, str) and s:
            lock = object.__getattribute__(self, "_partial_lock")
            try:
                with lock:
                    self._flush_partial_line(s)
            except Exception:
                pass
        return stream.write(s)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        # Flush any pending partial line so it becomes visible, then delegate.
        lock = object.__getattribute__(self, "_partial_lock")
        try:
            with lock:
                partial = object.__getattribute__(self, "_partial")
                if partial.strip():
                    object.__setattr__(self, "_partial", "")
                    self._emit_line(partial)
        except Exception:
            pass
        return object.__getattribute__(self, "_stream").flush()


# Module-level singleton + install state -------------------------------------
capture = ConsoleCapture()

_installed = False
_installed_lock = threading.Lock()
_prev_excepthook = None
_prev_threading_excepthook = None
_log_handler = None
_console_fallback_handler = None


def _excepthook(exc_type, exc_value, tb):
    try:
        capture.add_exception(exc_type, exc_value, tb)
    except Exception:
        pass
    prev = _prev_excepthook
    if prev is not None:
        try:
            prev(exc_type, exc_value, tb)
        except Exception:
            pass


def _threading_excepthook(args):
    try:
        capture.add_exception(args.exc_type, args.exc_value, args.exc_traceback)
    except Exception:
        pass
    prev = _prev_threading_excepthook
    if prev is not None:
        try:
            prev(args)
        except Exception:
            pass


def install(max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES):
    """Activate capture. Idempotent. Safe to call more than once."""
    global _installed, _prev_excepthook, _prev_threading_excepthook, _log_handler, _console_fallback_handler
    with _installed_lock:
        if _installed:
            return capture
        capture._max_lines = max(1, int(max_lines))
        capture._max_bytes = max(1024, int(max_bytes))

        orig_stderr = sys.stderr

        _log_handler = _CaptureLogHandler(capture)
        logging.getLogger().addHandler(_log_handler)

        # If the root logger has no other handlers, logging output previously reached the
        # console via the handler-of-last-resort — which stops firing once any handler exists.
        # Attach an explicit StreamHandler so the real console keeps working. Bound to the
        # pre-proxy stderr so records are not double-captured through the tee below.
        # (When --loglevel is set, setup_logging already added a real handler: skip this.)
        if not any(h is not _log_handler for h in logging.getLogger().handlers):
            _console_fallback_handler = logging.StreamHandler(orig_stderr)
            logging.getLogger().addHandler(_console_fallback_handler)

        if not isinstance(sys.stdout, _TeeStream):
            sys.stdout = _TeeStream(capture, sys.stdout, "stdout", "INFO")
        if not isinstance(sys.stderr, _TeeStream):
            sys.stderr = _TeeStream(capture, sys.stderr, "stderr", "ERROR")

        _prev_excepthook = sys.excepthook
        sys.excepthook = _excepthook
        _prev_threading_excepthook = threading.excepthook
        threading.excepthook = _threading_excepthook

        _installed = True
    return capture


def uninstall():
    """Deactivate capture (used by tests)."""
    global _installed, _prev_excepthook, _prev_threading_excepthook, _log_handler, _console_fallback_handler
    with _installed_lock:
        if not _installed:
            return
        if _log_handler is not None:
            try:
                logging.getLogger().removeHandler(_log_handler)
            except Exception:
                pass
            _log_handler = None
        if _console_fallback_handler is not None:
            try:
                logging.getLogger().removeHandler(_console_fallback_handler)
            except Exception:
                pass
            _console_fallback_handler = None
        if isinstance(sys.stdout, _TeeStream):
            sys.stdout = sys.stdout._stream
        if isinstance(sys.stderr, _TeeStream):
            sys.stderr = sys.stderr._stream
        if _prev_excepthook is not None:
            sys.excepthook = _prev_excepthook
            _prev_excepthook = None
        if _prev_threading_excepthook is not None:
            threading.excepthook = _prev_threading_excepthook
            _prev_threading_excepthook = None
        _installed = False
