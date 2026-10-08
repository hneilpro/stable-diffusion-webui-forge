# Research: console error visibility in stable-diffusion-webui-forge (TASK-003)

## What exists today
- `modules/errors.py` keeps `exception_records` (cap 5, no timestamps) of exceptions recorded via
  `errors.report()` / `errors.record_exception()`. Only consumer: `modules/sysinfo.py` ("Exceptions" in
  system-info dump). `modules/shared_state.py:182` also calls `record_exception()`.
- API error path: the live handler is `webui.py::_handle_exception` (wired via gradio
  `app_kwargs={"exception_handlers": {Exception: _handle_exception}}` in `webui_worker`, and directly in
  `api_only_worker`). It returns `{"error", "detail", "body", "message": str(e)}` — **no traceback in the
  response**. NOTE: `modules/api/api.py::handle_exception` (inside `api_middleware()`) is dead code: the
  `api_middleware(self.app)` call in `Api.__init__` is commented out with a FIXME, so that whole function
  never runs. Do not edit it.
- `webui.py::_handle_exception` (api-only mode): same — `str(e)` only.
- Logging: `modules/logging_config.py::setup_logging` attaches a root StreamHandler only when
  `SD_WEBUI_LOG_LEVEL` / `--loglevel` is set; otherwise Python default (WARNING to stderr). Much webui
  output is plain `print()`.
- API auth: `Api.add_api_route` adds `Depends(self.auth)` (HTTPBasic against `--api-auth`) when set;
  otherwise routes are open — same as every other route. No per-route auth to design.
- UI tabs: `modules/ui.py::create_ui` builds `interfaces = [(blocks, label, id), ...]`; adding one more
  entry creates a tab. Polling pattern available via `demo.load(..., every=N)`.

## What is missing
1. No unified capture of stdout/stderr/logging in-process.
2. No API exposure of console output or of the (already existing) structured exceptions.
3. The structured exception store is too small (5), timestamp-less, and bypassed by the API error handler.
4. No UI surface for the log.

## Prior art checked
- A1111 upstream has no console-log API; some forks tail a log file. We will not write a log file
  (non-goal); an in-memory ring buffer is cheaper and needs no disk/rotation design.
- `logging.handlers.MemoryHandler` exists but flushes on capacity to a target; a plain
  `collections.deque(maxlen=N)` with our own handler is simpler and gives cursor pagination directly.

## Load-bearing facts (verified in tree)
- `handle_exception` is reached for API errors via both middleware and `@app.exception_handler(Exception)`.
- `Api.__init__` defines all routes through `self.add_api_route`, so new routes inherit auth automatically.
- `webui.py` top-level runs before `initialize.imports()` — earliest safe install point for capture.
- `modules/cmd_args.py` holds the argparse parser (`--loglevel` precedent at line 21).
