# TASK-003 critic review (self-review — not independent; reviewer built the change)

## Verdict: needs changes (1 blocking)

## Rubric

### 1. Research — PASS with note
Right questions asked; primary sources read (errors.py, api.py, webui.py, logging_config.py, cmd_args.py).
Load-bearing correction made mid-build: `api.py::handle_exception` is dead code (api_middleware call
commented out) — verified by grep, edit reverted, ADR/research updated. Note: the `demo.load(every=)`
and `show_copy_button` claims were verified after the fact against the gradio 4.40.0 wheel
(`every` is a standard event param in events.py; Textbox has `show_copy_button` and `autoscroll`) —
both check out, but they should have been verified before build.

### 2. Plan/Design — PASS
ADR-003 exists, alternatives real (file tailing, MemoryHandler, dup2 all rejected with reasons),
auth inheritance explicit, bounds explicit. Kill switch included.

### 3. Build — FAIL (blocking finding F1)
Diff scoped (7 files, +132/-6), no secrets, style respected. But:

**F1 — BLOCKING: default-config console regression.** `install()` unconditionally adds
`_CaptureLogHandler` to the root logger. CPython only uses the handler-of-last-resort (which is what
currently prints `logging` output to the console in the webui process) when NO handlers are found.
`setup_logging()` runs only in the launcher process (`launch_utils.py:26`; `start()` does
`import webui` in-process), and with no `--loglevel` it returns early — so in the default
configuration the webui process has zero root handlers and logging reaches the console via lastResort.
After this change, every `logging.warning/error/...` would be swallowed into the buffer and never
printed to the real console. Reproduced the semantics in isolation:
`logging.getLogger("x").warning()` prints with no handlers, goes silent after adding a no-op handler.
Fix: in `install()`, if the root logger has no other handlers, also attach a plain
`logging.StreamHandler` bound to the pre-proxy stderr (bound pre-proxy to avoid double-capture via
the tee); remove it in `uninstall()`. Add a test asserting a `logging.warning` lands both in the
buffer and on the original stderr.

**F2 — Minor:** `_TeeStream._partial` is updated without a lock; concurrent `end=""` writes from
multiple threads can garble/drop a partial line. Cosmetic only (complete `print()` lines are single
`write()` calls and unaffected). Consider a lock in `write()`.

**F3 — Minor:** `webui.py::_handle_exception` records even `HTTPException`s (e.g. 401 auth probes)
into the 50-slot structured store — noise that can flush real tracebacks. Suggest skipping
`HTTPException` there, mirroring the original "do not print backtrace on known httpexceptions" rule.

### 4. Test — PASS with note
21/21 pytest pass, all meaningful (bounds, cursor pagination, thread-safety with unique monotonic
IDs — the ID-under-lock bug was caught by the test, good). py_compile 8/8. Handler logic verified
via AST-extract against stub models with the real buffer. Skips correctly labelled `unverified`
(live HTTP, UI tab render, real error surfacing — need owner PC). Note: the F1 regression above is
exactly the class of bug offline tests didn't cover (no test asserted console preservation); the
repair must add one.

### 5. Correctness — conditional
Acceptance criteria 1–4 verified on evidence at the handler/buffer level; criterion 5 (UI tab) and
the HTTP transport are unverified by construction here. Criterion 4's "never raises" and bounds hold.
F1 must be repaired before any done claim.

### 6. Evidence quality — PASS
A stranger can reproduce: test report lists exact commands, exit statuses, and what was stubbed.

### 7. Risk
Biggest remaining risk: the early `import webui` path on the owner PC (arguably low — the added import
is a rebinding of an already-imported module since `modules/shared.py` imports `shared_cmd_options`).
Auth: routes inherit `--api-auth` behavior; without it the log is open like every other route —
disclosed in ADR, consistent with existing threat model. C-extension (torch/CUDA) writes still bypass
capture — documented non-goal.

### 8. Suggestions
- (opinion) Consider `autoscroll=True` on the console textbox (verified present in 4.40.0) so the
  newest line stays visible during the 2s poll.
- (opinion) The `since` linear scan is fine at 2000 entries; no index needed.

## Next
Builder repairs F1 (+ regression test), optionally F2/F3, then critic re-scores F1–F3 only. Live
4090 checks (HTTP + UI tab + real error surfacing) remain `unverified` until the owner runs them.

## Re-review (2026-10-08) — F1–F3 re-score

**F1 — FIXED.** `install()` now attaches a `logging.StreamHandler` bound to the pre-proxy stderr when
the root logger has no other handlers, and `uninstall()` removes it. Verified firsthand: read the
final `install()`/`uninstall()`; the new regression test fails on the pre-fix hunk (builder proved via
revert: 1 failed) and passes now; full suite 24/24 green in this pass; py_compile clean. The skip-when-
`--loglevel` branch is covered by `test_install_with_existing_handler_no_duplicate_console`.
**F2 — FIXED.** `_partial_lock` added; `write()`/`flush()` hold it around the read-modify-write;
`_flush_partial_line` itself takes no lock (no double-lock); no lock-ordering cycle (`capture.add`
never calls back into the tee). Covered by `test_tee_proxy_concurrent_partial_writes`.
**F3 — FIXED.** `webui.py::_handle_exception` uses `isinstance(e, HTTPException)` (imported from
`fastapi.exceptions`) to skip known HTTP errors in the structured store; console/response behavior
unchanged.

## Verdict: approve with notes (self-review — not independent)
The offline core is sound and the blocking regression is repaired with a proven regression test.
Notes: (1) live HTTP + UI-tab + real-error surfacing remain `unverified` — owner PC run needed
(pull branch, restart Forge, tunnel URL, then hit both endpoints and break something on purpose);
(2) `--disable-console-log-capture` flag parsing itself is unverified here (cmd_args imports the
heavy launch chain); (3) C-extension writes still bypass capture by design. No secrets in the diff;
rollback is `git revert` of the 8 files (plus deleting the 2 new files). Not pushing — owner reviews
at push time per repo rules.
