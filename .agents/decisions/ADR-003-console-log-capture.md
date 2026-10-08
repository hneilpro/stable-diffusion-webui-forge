# ADR-003 — In-process console capture exposed over API and UI

## Context
Remote debugging over a tunnel is blind: API 500s carry only `str(e)`, the traceback exists only on the
server console. We need the console (stdout/stderr/logging/tracebacks) queryable over the authenticated
API and visible in a UI tab.

## Decision
Build a small in-process capture layer, no log files, no new dependencies:

1. **New `modules/console_capture.py` (stdlib only)** owns a `ConsoleCapture` singleton:
   - `collections.deque(maxlen=2000)` of entries `{id, ts, stream, level, text}`; monotonic `id` cursor.
     Also a soft byte cap (~512 KB) — when exceeded, oldest lines are dropped first.
   - A `logging.Handler` attached to the **root logger** at install time (level NOTSET) so every record
     that reaches root is captured regardless of `SD_WEBUI_LOG_LEVEL`. Plus a plain
     `logging.StreamHandler` bound to the pre-proxy stderr when root has no other handlers —
     otherwise the handler-of-last-resort stops firing and `logging` output would vanish from the
     real console in the default (no `--loglevel`) configuration. Skipped when `--loglevel` set
     `setup_logging`'s own handler (no double-print).
   - A tee proxy for `sys.stdout`/`sys.stderr`: `write`/`writelines` append to the buffer (split on
     newlines, `level` inferred: `ERROR` for stderr, `INFO` for stdout) then delegate to the original
     stream. All other attributes delegate via `__getattr__` so `.buffer`/`.fileno`/`.isatty()` keep
     working. Multi-line writes are split; partial lines are held until a newline (avoids tqdm-style
     fragments flooding the buffer). `\r`-only progress rewrites are coalesced: a line that starts with
     `\r` replaces the current partial line rather than appending.
   - `sys.excepthook` / `threading.excepthook` install that formats the full traceback into the buffer
     (and into `modules.errors` structured store) before chaining to the previous hook.
   - `emit`/`write` never raise: everything is wrapped in try/except with a re-entrancy guard, so a
     capture failure can never break the app or recurse.
   - Read API: `tail(limit)` / `since(cursor, limit)` returning `(new_cursor, [entries])`.
2. **Install in `webui.py` immediately after imports**, before `initialize_forge()` — earliest point where
   our module is importable. Both `api_only_worker` and `webui_worker` paths inherit it.
   Kill switch: `--disable-console-log-capture` in `modules/cmd_args.py` (default: capture ON).
3. **API (`modules/api/api.py`)**: 
   - `GET /sdapi/v1/console-log?limit=200&since=0` → `{cursor, lines}` (pydantic models in
     `modules/api/models.py`). `limit` clamped to [1, 2000].
   - `GET /sdapi/v1/console-log/exceptions` → `{exceptions: [...]}` from `errors.get_exceptions()`.
   - Both registered via `self.add_api_route` → identical auth behavior to every other route (HTTPBasic
     when `--api-auth` is set).
4. **`modules/errors.py`**: add ISO-8601 timestamp to each record, raise cap 5 → 50, dedupe fixed
   (the old check compared a dict to an exception object, so it never fired). One-line addition:
   `errors.record_exception_info(type(e), e, e.__traceback__)` inside `webui.py::_handle_exception` —
   the single live API error handler (used both by the gradio-launched app via `app_kwargs` and by
   `api_only_worker`). Note: `modules/api/api.py::handle_exception` inside `api_middleware()` is dead
   code — its call is commented out with a FIXME — so it was deliberately left untouched.
5. **UI (`modules/ui.py`)**: append a `gr.Blocks` "Console" tab to `interfaces`: a read-only textbox
   (monospace, 40 lines), Refresh button, auto-refresh checkbox; `demo.load(fn=..., every=2)` polls the
   buffer server-side and updates the textbox. Bounded work per poll: one buffer slice + string join.

## Alternatives rejected
- Tailing a log file: requires a file, rotation, and a reader — more moving parts for zero gain; the
  buffer already answers "what just happened".
- `logging.handlers.MemoryHandler`: flush-on-capacity semantics don't fit cursor pagination.
- fd-level (`os.dup2`) capture of C-extension output: invasive (breaks debuggers/profilers), out of scope.
- Scrubbing secrets from log lines: unreliable (can't enumerate all secret shapes); instead the routes
  inherit API auth and the docs note that console output may contain paths/hostnames.

## Consequences
- ~+200 lines in `modules/console_capture.py`, small diffs in 6 existing files, new tests alongside.
- Memory: bounded by construction (2000 lines + 512 KB cap).
- Perf: one string format + deque append per log line; negligible vs. model work.
- Live proof (routes + UI tab + a real error surfacing) still requires the owner PC run — recorded as
  `unverified` in the test report until then.
