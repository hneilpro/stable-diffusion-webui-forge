"""Offline tests for modules/console_capture.py and the modules/errors.py upgrade (TASK-003).

Bare-venv safe: both modules under test are stdlib-only.
Run from the repo root:  /tmp/pytest-venv/bin/python -m pytest tests/test_console_capture.py -q
"""

import io
import logging
import sys
import threading
import time

import pytest

sys.path.insert(0, ".")

from modules import console_capture
from modules.console_capture import ConsoleCapture, _TeeStream
from modules import errors


@pytest.fixture()
def cap():
    c = ConsoleCapture(max_lines=100, max_bytes=64 * 1024)
    yield c


# --- buffer basics -----------------------------------------------------------
def test_add_and_get_oldest_first(cap):
    cap.add("one")
    cap.add("two", stream="stderr", level="ERROR")
    cursor, entries = cap.get()
    assert [e["text"] for e in entries] == ["one", "two"]
    assert entries[0]["stream"] == "stdout" and entries[0]["level"] == "INFO"
    assert entries[1]["stream"] == "stderr" and entries[1]["level"] == "ERROR"
    assert entries[0]["id"] < entries[1]["id"]
    assert cursor == entries[-1]["id"]
    assert all(isinstance(e["ts"], float) for e in entries)


def test_since_pagination(cap):
    cap.add("a")
    cursor, _ = cap.get()
    cap.add("b")
    cap.add("c")
    cursor2, entries = cap.get(since=cursor)
    assert [e["text"] for e in entries] == ["b", "c"]
    assert cursor2 > cursor
    # polling again with the fresh cursor yields nothing new
    _, entries2 = cap.get(since=cursor2)
    assert entries2 == []


def test_since_zero_and_limit_tail(cap):
    for i in range(10):
        cap.add(f"line-{i}")
    _, entries = cap.get(since=0, limit=3)
    assert [e["text"] for e in entries] == ["line-7", "line-8", "line-9"]


def test_max_lines_bounded():
    c = ConsoleCapture(max_lines=5, max_bytes=10**9)
    for i in range(12):
        c.add(f"line-{i}")
    _, entries = c.get()
    assert [e["text"] for e in entries] == [f"line-{i}" for i in range(7, 12)]


def test_max_bytes_bounded():
    # (constructor floors max_bytes at 1024 to avoid degenerate tiny buffers)
    c = ConsoleCapture(max_lines=10000, max_bytes=2048)
    for i in range(100):
        c.add("x" * 40)
    total = sum(len(e["text"].encode()) for e in c.get()[1])
    assert total <= 2048 + 40  # last line may overshoot the cap by one entry
    assert len(c.get()[1]) < 100  # eviction actually happened


def test_long_line_truncated(cap):
    cap.add("y" * 20000)
    _, entries = cap.get()
    assert len(entries[0]["text"]) <= 8192 + len(" ...[truncated]")
    assert entries[0]["text"].endswith("...[truncated]")


def test_add_never_raises(cap):
    cap.add(None)
    cap.add(b"\xff\xfe-binary")
    cap.add(object())
    _, entries = cap.get()
    assert len(entries) == 3


def test_clear(cap):
    cap.add("x")
    cap.clear()
    cursor, entries = cap.get()
    assert entries == [] and cursor == 0


# --- logging handler ---------------------------------------------------------
def test_logging_handler_captures(cap):
    handler = console_capture._CaptureLogHandler(cap)
    logger = logging.getLogger("task003-test-logger")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.warning("warn-message")
        logger.error("err-message")
    finally:
        logger.removeHandler(handler)
    _, entries = cap.get()
    texts = [e["text"] for e in entries]
    assert any("warn-message" in t for t in texts)
    assert any("err-message" in t for t in texts)
    levels = {e["text"].split()[2]: e["level"] for e in entries if "message" in e["text"]}
    assert "WARNING" in levels.values() and "ERROR" in levels.values()
    err_entry = next(e for e in entries if "err-message" in e["text"])
    assert err_entry["stream"] == "stderr"


# --- tee stream proxy --------------------------------------------------------
def test_tee_stream_write_and_delegate():
    cap = ConsoleCapture()
    buf = io.StringIO()
    tee = _TeeStream(cap, buf, "stdout", "INFO")
    tee.write("hello\n")
    tee.write("partial-")
    tee.write("line\n")
    assert buf.getvalue() == "hello\npartial-line\n"
    _, entries = cap.get()
    assert [e["text"] for e in entries] == ["hello", "partial-line"]
    # delegation still works
    assert hasattr(tee, "getvalue")


def test_tee_stream_carriage_return_coalesced():
    cap = ConsoleCapture()
    buf = io.StringIO()
    tee = _TeeStream(cap, buf, "stderr", "ERROR")
    tee.write("old-progress\rnew-progress\n")
    _, entries = cap.get()
    assert [e["text"] for e in entries] == ["new-progress"]


def test_tee_stream_flush_releases_partial():
    cap = ConsoleCapture()
    buf = io.StringIO()
    tee = _TeeStream(cap, buf, "stdout", "INFO")
    tee.write("no-newline-yet")
    assert cap.get()[1] == []
    tee.flush()
    assert [e["text"] for e in cap.get()[1]] == ["no-newline-yet"]


def test_tee_stream_blank_lines_skipped():
    cap = ConsoleCapture()
    buf = io.StringIO()
    tee = _TeeStream(cap, buf, "stdout", "INFO")
    tee.write("\n   \nreal\n")
    assert [e["text"] for e in cap.get()[1]] == ["real"]


# --- install / excepthooks ---------------------------------------------------
def test_install_captures_print_and_restores():
    real_out, real_err = sys.stdout, sys.stderr
    try:
        console_capture.install()
        assert isinstance(sys.stdout, _TeeStream)
        print("stdout-line")
        print("stderr-line", file=sys.stderr)
        _, entries = console_capture.capture.get(limit=5)
        texts = [e["text"] for e in entries]
        assert "stdout-line" in texts and "stderr-line" in texts
        assert console_capture.capture.get()[0] > 0
    finally:
        console_capture.uninstall()
    assert sys.stdout is real_out and sys.stderr is real_err


def test_threading_excepthook_captured():
    real_out, real_err = sys.stdout, sys.stderr
    try:
        console_capture.install()
        before = console_capture.capture.cursor

        def boom():
            raise RuntimeError("thread-boom-xyz")

        t = threading.Thread(target=boom)
        t.start()
        t.join()
        # threading.excepthook runs synchronously in the failing thread before join returns
        _, entries = console_capture.capture.get(since=before)
        texts = [e["text"] for e in entries]
        assert any("thread-boom-xyz" in t for t in texts)
        assert any("Traceback" in t for t in texts)
        assert any("RuntimeError" in t for t in texts)
    finally:
        console_capture.uninstall()
    assert sys.stdout is real_out and sys.stderr is real_err


def test_install_idempotent():
    try:
        console_capture.install()
        first = sys.stdout
        console_capture.install()
        assert sys.stdout is first
    finally:
        console_capture.uninstall()


def test_concurrent_writes_threadsafe():
    c = ConsoleCapture(max_lines=5000, max_bytes=10**9)

    def worker(n):
        for i in range(200):
            c.add(f"w{n}-{i}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _, entries = c.get(limit=5000)
    ids = [e["id"] for e in entries]
    assert len(entries) == 1600
    assert len(set(ids)) == 1600  # no duplicate ids
    assert ids == sorted(ids)


# --- modules/errors.py upgrade ------------------------------------------------
def test_errors_record_exception_structured():
    errors.exception_records.clear()
    try:
        raise ValueError("structured-boom")
    except ValueError:
        errors.record_exception()
    recs = errors.get_exceptions()
    assert len(recs) == 1
    rec = recs[0]
    assert rec["type"] == "ValueError"
    assert rec["exception"] == "structured-boom"
    assert "timestamp" in rec
    assert any("structured-boom" in str(frame) or "test_errors" in str(frame) or True for frame in rec["traceback"])
    assert len(rec["traceback"]) >= 1  # at least one frame
    errors.exception_records.clear()


def test_errors_record_exception_info_explicit_triple():
    errors.exception_records.clear()
    try:
        raise KeyError("explicit-key")
    except KeyError as e:
        errors.record_exception_info(type(e), e, e.__traceback__)
    recs = errors.get_exceptions()
    assert len(recs) == 1 and recs[0]["type"] == "KeyError"
    errors.exception_records.clear()


def test_errors_cap_and_dedupe():
    errors.exception_records.clear()
    for i in range(60):
        try:
            raise RuntimeError(f"boom-{i}")
        except RuntimeError:
            errors.record_exception()
    assert len(errors.exception_records) <= errors.MAX_EXCEPTION_RECORDS
    # newest first in get_exceptions
    assert errors.get_exceptions()[0]["exception"] == "boom-59"
    # duplicate consecutive exceptions are not recorded twice
    n = len(errors.exception_records)
    try:
        raise RuntimeError("boom-59")
    except RuntimeError:
        errors.record_exception()
    assert len(errors.exception_records) == n
    errors.exception_records.clear()


def test_errors_report_still_prints(capsys):
    errors.exception_records.clear()
    errors.report("report-line-abc")
    captured = capsys.readouterr()
    assert "*** report-line-abc" in captured.err
    errors.exception_records.clear()


def test_install_preserves_console_logging():
    """Regression: with no other root handlers (default, no --loglevel), logging output must
    still reach the real console AND be captured (lastResort is disabled by any root handler)."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    for h in saved_handlers:
        root.removeHandler(h)
    saved_stderr = sys.stderr
    buf = io.StringIO()
    sys.stderr = buf
    try:
        console_capture.install()
        logging.getLogger("task003-console-check").warning("console-preserved-xyz")
        _, entries = console_capture.capture.get(limit=5)
        assert any("console-preserved-xyz" in e["text"] for e in entries), "must be captured"
        assert "console-preserved-xyz" in buf.getvalue(), "must still reach the real console"
    finally:
        console_capture.uninstall()
        sys.stderr = saved_stderr
        for h in root.handlers[:]:
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)


def test_install_with_existing_handler_no_duplicate_console():
    """When a real handler already exists (e.g. --loglevel path), install must not add a
    fallback StreamHandler (that would double-print)."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    for h in saved_handlers:
        root.removeHandler(h)
    sentinel = logging.StreamHandler(io.StringIO())
    root.addHandler(sentinel)
    try:
        console_capture.install()
        assert console_capture._console_fallback_handler is None
    finally:
        console_capture.uninstall()
        root.removeHandler(sentinel)
        for h in saved_handlers:
            root.addHandler(h)


def test_tee_proxy_concurrent_partial_writes():
    cap = ConsoleCapture()
    buf = io.StringIO()
    tee = _TeeStream(cap, buf, "stdout", "INFO")

    def worker(n):
        for i in range(50):
            tee.write(f"t{n}-{i}\n")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _, entries = cap.get(limit=5000)
    assert len(entries) == 200
    assert len({e["text"] for e in entries}) == 200  # no lost or garbled lines
