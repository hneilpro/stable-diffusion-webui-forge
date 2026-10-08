# TASK-003 handoff — console errors/logs over API and UI

## What changed
- New `modules/console_capture.py` (stdlib-only): in-process ring buffer (2000 lines / 512 KB) fed by
  a root-logging handler, sys.stdout/stderr tee proxy, and sys/threading excepthooks. Cursor-based
  pagination. Never raises into the app; `--disable-console-log-capture` kill switch.
- `webui.py`: installs capture right after imports (before `initialize_forge()`); `_handle_exception`
  (the single live API error handler, both app paths) records structured exceptions, skipping
  `HTTPException`s.
- `modules/api/api.py`: `GET /sdapi/v1/console-log?limit&since` and
  `GET /sdapi/v1/console-log/exceptions`, via `add_api_route` (same API auth as all routes).
- `modules/api/models.py`: `ConsoleLogLine`, `ConsoleLogResponse`, `ConsoleExceptionsResponse`.
- `modules/errors.py`: records gain `type` + UTC `timestamp`; cap 5 → 50; dedupe actually works now;
  new `record_exception_info(exc_type, exc_value, tb)` for handler contexts.
- `modules/cmd_args.py`: `--disable-console-log-capture`.
- `modules/ui.py` + `style.css`: "Console" tab (newest-first, monospace), Refresh/Clear buttons,
  2s auto-poll via `demo.load(every=2)`.
- `tests/test_console_capture.py`: 24 tests, all passing.

## What was verified
- 24/24 pytest (buffer, pagination, bounds, proxy, excepthooks, thread-safety, console-preservation
  regression test proven to fail pre-fix), py_compile 8/8, handler logic via AST-exec against stubs.
- Critic review: 1 blocking found (console-output regression in default config) → repaired with
  fallback StreamHandler + regression test → re-review: approve with notes (self-review).

## What is NOT verified (needs owner PC)
- Endpoints responding over HTTP with auth; Console tab rendering; a real error (e.g. poisoned Flux
  state) surfacing in both the log buffer and structured exceptions. Pull branch, restart Forge,
  paste tunnel URL, then verify live.

## Risks / notes
- `modules/api/api.py::handle_exception` (inside `api_middleware`) is dead code — deliberately untouched.
- Without `--api-auth`, the log endpoints are open like every other route (documented in ADR-003).
- C-extension (torch/CUDA) fd-level writes bypass capture by design.
