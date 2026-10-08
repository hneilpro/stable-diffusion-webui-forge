# TASK-003 — Console errors/logs over API and UI

## Goal
Give the Forge webui a way to expose console output (errors, tracebacks, log lines) over the authenticated REST API and in a small UI tab, so remote debugging over a tunnel no longer requires access to the PC's console window. (Motivation: API 500s return only `str(e)`; the real traceback lives only on the server console.)

## Non-goals
- fd-level capture of C-extension (torch/CUDA) writes — Python-level only.
- Remote log shipping, log file persistence, or log rotation.
- Any change to API auth; new routes reuse the existing auth dependency.
- Fixing the Flux VAE loader (separate thread).

## Acceptance criteria
1. `GET /sdapi/v1/console-log?limit=N&since=C` returns `{cursor, lines:[{id, ts, stream, level, text}]}`; incremental polling works (`since` = last cursor returns only newer lines).
2. `GET /sdapi/v1/console-log/exceptions` returns structured recent exceptions (type, message, formatted traceback, timestamp), including exceptions that previously only went to the rich-console printer in the API error handler.
3. Both routes are behind the same API auth as all other `/sdapi` routes.
4. Buffer is bounded (default 2000 lines / ~512 KB), thread-safe, never raises into the app, and can be disabled with `--disable-console-log-capture`.
5. A "Console" UI tab shows the log with refresh + auto-refresh (2s poll).
6. All new core logic (capture buffer, proxy, excepthook, pagination) covered by offline pytest; py_compile clean on every touched file.

## Checks
- `python -m py_compile` on all touched/new files — exit 0.
- `pytest` on new tests — all pass.
- Import purity: `modules/console_capture.py` imports stdlib only.
- Live 4090 proof (routes respond, UI tab renders, real render error appears in log): needs owner PC — mark `unverified` until run.

## Result
(pending)
