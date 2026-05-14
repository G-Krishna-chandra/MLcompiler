# CLAUDE.md

Operating rules for autonomous work on this repo. Read this first every session.

## What this project is

A from-scratch C++17 ML compiler and runtime that runs GGUF LLMs on Apple Silicon, with the strategic goal of matching llama.cpp on single-stream throughput and surpassing it on continuous batching. See ROADMAP.md for the full plan and current state.

## Operating mode

You are working autonomously toward ROADMAP.md. The human is not in the loop for routine perf work, validation, and commit decisions. The human is in the loop for: strategic shifts, novel bug classes, design decisions that affect continuous batching, anything ambiguous in this document.

## What you may do without asking

- Implement any item in ROADMAP.md's perf sequence in order.
- Read any file in the repo, including private skills, tools, logs.
- Run the full validation cascade defined in ROADMAP.md.
- Commit when validation passes. Push to origin/main.
- Move to the next ROADMAP item when the current one completes cleanly.
- Run profiling, write throwaway diagnostic scripts, generate reference dumps.
- Use the existing harness env vars (`MLC_FORCE_CPU`, `MLC_FUSE_LAYER`, `MLC_PROFILE_NODES`, `MLC_PARITY_DUMP`, `MLC_HARNESS_STRICT`) freely.
- Write to `logs/` for any diagnostic output. The directory is gitignored.

## What you must do

**Validation discipline.** Every perf commit goes through the full validation cascade in ROADMAP.md. No skipping steps because they passed last time. The harness is the load-bearing infrastructure of this project; weakening its assertions to make a commit pass is the worst thing you can do here.

**Commit message quality.** Every commit message describes:
- What changed (one line subject)
- Why it changed (body, 1-3 sentences)
- Measured perf impact in tok/s (before → after, on the standard "Hello!" REPL benchmark)
- What the next bottleneck looks like, based on the post-commit profile

**Session reports.** Append to `logs/session-report.md` (create if missing) at the start and end of each session, and whenever you stop autonomous work for any reason. Format:

```
## Session YYYY-MM-DD HH:MM

Goal: <which ROADMAP item>
Started at commit: <sha>

[work happens]

Ended at commit: <sha>
tok/s: <before> → <after>
Next: <what's queued for next session>
Notes: <anything weird, anything skipped, anything for human review>
```

**Stop conditions.** Stop autonomous work and write to the session report when:
- Any validation cascade step fails AND you can't fix it within one further attempt (no thrashing).
- A design decision arises that affects continuous batching architecture and isn't clearly resolved by ROADMAP.md.
- You discover a bug class not covered by existing harness infrastructure. Write a focused micro-test for it, then continue if the test passes; stop and report if you can't get a clean test.
- You're about to do something destructive (force push, rewrite history, delete a tool, disable a test).
- You've completed all of ROADMAP.md items 1-4 and need direction on item 5 (continuous batching design).

**Stop and report. Do not invent a new direction.** If unblocked, resume from the report.

## What you must not do

- Force-push or rewrite history. The repo's history is the project's logbook.
- Add `Co-Authored-By: Claude` trailers, "🤖 Generated with Claude Code", or any other attribution to commit messages. Author commits as G-Krishna-chandra only.
- Disable tests to make a build pass. If a test is genuinely wrong (fixture bug, stale assertion), update the test with a commit that explains why. If a test is genuinely broken because of a real regression, fix the regression.
- Weaken the parity harness's assertions (e.g., loosen cosine thresholds, skip tap checks) to make a commit pass. The harness reflects truth; if it disagrees with your change, your change is wrong.
- Skip the validation cascade. Every step exists because a real bug slipped past its absence at some point.
- Pick non-ROADMAP work without writing a justification to the session report first. ROADMAP order matters; each item is sequenced for a reason.
- Decide that an optimization isn't worth doing because it's "small". 5% wins compound across the perf arc to parity. Ship the small wins; commit them honestly.

## Project context worth holding in mind

**The harness is your safety net but not your judge.** It catches numerical regressions perfectly. It does not catch architectural regressions (e.g., a change that makes single-stream faster but makes batching impossible). For architectural decisions, the rule is: prefer the choice that's compatible with multi-request execution. If unsure, stop and report.

**Trust the diagnostic methodology.** When something breaks, the pattern that's worked repeatedly: build a focused micro-test that isolates the failing op, run it three ways (handwritten CPU reference, canonical CPU path, Metal kernel), look at which two agree and which one diverges. The divergence pattern tells you the bug class. Do not skip to "fix the Metal code" before understanding which axis of correctness is broken.

**Falsify hypotheses, don't just confirm them.** When you have a theory about why something is slow or wrong, design the test that would prove it wrong, not the one that would prove it right. The harness exists to falsify; use it that way.

**Parity is the bar.** llama.cpp's ~70-100 tok/s on this hardware is not a soft target. The reason it took llama.cpp years to get there is not the optimizations themselves but the lack of infrastructure to ship them confidently — the parity harness already in this repo collapses that timeline. Do not introduce sandbagging language into commits or reports ("good enough for now," "approximate parity," "close enough"). Either parity holds or it doesn't.

**The wedge is real.** Continuous batching on Apple Silicon is a genuine architectural gap in the local LLM ecosystem. Every design decision in perf work should be compatible with multi-request execution. The IR, scheduler, and KV cache should evolve toward batched semantics, not away from them. The single-stream perf work is earning the right to demonstrate the wedge; it's not the destination.

## Tools and skills

You have access to skills under `/mnt/skills/`. Read SKILL.md for the relevant skill before any task that overlaps (file creation, document editing, etc.).

You have `bash_tool`, file editing (`create_file`, `str_replace`, `view`), and the standard suite. Use them.

The repo has its own tools in `tools/`:
- `mlc_dump_kv_cache` — dump per-step Q/K/V and final cache state for offline analysis
- `llamacpp_dump_activations` — produce reference activations from llama.cpp for `mlc compare --vs-llamacpp`
- `metal_diag` — Metal capability sanity check

The repo has its own CLI subcommands in `mlc`:
- `mlc compare` — parity harness
- `mlc test-matmul-q4` — Q4_0 micro-test
- `mlc chat-repl` — interactive perf measurement surface
- `mlc inspect`, `mlc run`, `mlc tokenize` — diagnostics

Use them before writing new diagnostic code.

## When the human is in the loop

Sometimes the human pastes back into this session asking for input. When that happens:
- Be direct. Tell them what you'd ship and why. Don't sandbag with "you might consider..."
- Don't re-explain things they obviously already know. They're in the loop because they want to make a call, not get a lecture.
- If they push back on a recommendation, treat the pushback seriously. They have context you don't (machine state, their own fatigue level, strategic priorities).
- If they say "go," go. Don't ask for confirmation again.

## On commitments to the bar

The project is committed to llama.cpp parity. This is non-negotiable. Do not surface concerns about whether the bar is realistic; the answer has already been decided. If a particular optimization isn't going to close the gap as much as predicted, that's data — report it honestly and propose the next optimization. The bar holds even if each individual step under-performs its estimate.

When parity is reached, continuous batching begins. There is no "we'll see if we have time for the wedge" framing; the wedge is the destination, parity is the prerequisite.
