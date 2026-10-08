# TASK-003 offline test report

## Environment
Bare venv `/tmp/pytest-venv` (pytest 9.1.1, stdlib only). Repo at `/tmp/forge-repo`,
branch `feat/forge-consistent-character`. No GPU, no gradio/fastapi/pydantic — anything needing
them is marked unverified below, never a pass.

## Ran
- `python -m py_compile` on all touched/new files: **exit 0, 8/8**
  (modules/console_capture.py, modules/errors.py, modules/api/api.py, modules/api/models.py,
  modules/cmd_args.py, webui.py, modules/ui.py, tests/test_console_capture.py)
- `pytest tests/test_console_capture.py`: **21 passed** (buffer basics, since/cursor pagination,
  max_lines + max_bytes bounds, line truncation, never-raises, clear, logging-handler capture with
  level→stream mapping, tee-proxy write/delegation/CR-coalescing/flush/blank-skip, install/uninstall
  print capture + stream restore, threading excepthook end-to-end, install idempotency, 8-thread
  × 200-write thread-safety with unique monotonic IDs, errors.py structured record / explicit-triple
  record / 50-cap / dedupe / report-still-prints)
- API handler logic via AST-extract + stub-models exec (real `console_capture` + `errors`):
  route lines present via `self.add_api_route` (auth inherited), `console_log_api` pagination +
  cursor + empty-on-fresh-cursor verified, `console_log_exceptions_api` returns the recorded
  RuntimeError with type/message. **HANDLERS_OK**
- `webui.py::_handle_exception` patch: source-verified (fastapi not installed here); the added
  lines are try/except around the already-tested `record_exception_info`.
- Import purity: `modules/console_capture.py` imports stdlib only (verified by reading imports);
  `modules/errors.py` likewise.

## Bugs found and fixed during testing
1. ID assigned outside the buffer lock → concurrent appends could order out of ID sequence,
   breaking cursor pagination. Fixed: ID assigned under the lock (test `test_concurrent_writes_threadsafe`).
2. Test expectation wrong on `max_bytes` floor (constructor floors at 1024). Fixed the test.
3. Test used `get()` default limit=200 and asserted full count. Fixed the test.
4. Research correction: `modules/api/api.py::handle_exception` is dead code (`api_middleware` call
   commented out with FIXME). Reverted an edit there; the live handler is `webui.py::_handle_exception`
   (patched instead). ADR + research note updated.

## Unverified (needs owner PC + tunnel)
- `GET /sdapi/v1/console-log` and `/sdapi/v1/console-log/exceptions` respond over HTTP with auth.
- "Console" tab renders; 2s auto-refresh updates; Refresh/Clear buttons work.
- A real render failure (e.g. the poisoned Flux state) appears in both the log buffer and the
  structured exceptions with its traceback.
- `python webui.py` startup with the early `shared_cmd_options` import (no cycle expected —
  `modules/shared.py` already imports it — but not executed here).

## Repair round 1 (2026-10-08)
- F1: `install()` now attaches a plain `logging.StreamHandler` (bound to pre-proxy stderr, removed on
  `uninstall()`) when the root logger has no other handlers — console output preserved in the default
  no-`--loglevel` config; skipped when `setup_logging` already added one (no double-print).
- F2: `_TeeStream` partial-line buffer now guarded by a lock in `write()`/`flush()`.
- F3: `webui.py::_handle_exception` skips `HTTPException` (isinstance) for the structured store.
- New tests: `test_install_preserves_console_logging` (fails on pre-fix code — verified by reverting
  the hunk: 1 failed; passes on fixed code), `test_install_with_existing_handler_no_duplicate_console`,
  `test_tee_proxy_concurrent_partial_writes`. Suite: **24 passed**.
- Declined critic opinion `autoscroll=True`: the tab renders newest-first, so it is moot.
