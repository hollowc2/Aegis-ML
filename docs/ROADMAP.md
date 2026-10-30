# Aegis-ML Roadmap

The working plan for turning Aegis-ML from a demo into a firewall that runs in
production in front of cobblebob's LLM, opens that LLM to the public
safely, and then becomes a strong portfolio project built on that deployment.

**This file is the single source of truth across sessions.** Read it at the
start of any session that changes direction, ships a feature or finishes a
milestone. Update it in the same commit as the work.

## North star

> Aegis-ML is what lets anyone on the web talk to cobblebob's LLM safely, and
> the repo proves it with live numbers.

Every task should move toward that sentence. Work that doesn't (new classifier
research, more benchmark variants, more deployment targets) goes in the
[parking lot](#parking-lot).

## How to use this file

- Status markers: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped (say why)
- Work milestones in order. A milestone is done when its **Done when** line is true,
  not when every box is ticked; drop boxes that turn out not to matter.
- When something ships, tick it, add a dated line to the [changelog](#changelog),
  and record any choice made along the way under [decisions](#decisions).
- New ideas go in the parking lot, not into the current milestone.
- Work that lands in the cobblebob repo is marked **(Blockhead)**. Follow that
  repo's own deployment practice (hash manifests, backups, supervised rollout)
  for anything installed on the bot's host.

---

## The target

**cobblebob** is a Minecraft companion bot (separate repo, "Blockhead") with a
public web page. A local llama.cpp server runs its LLM on the same host.
Infrastructure specifics (host, ports, network, hardware) live in the gitignored
`docs/DEPLOYMENT.local.md`.

- cobblebob calls `GET /v1/models` and non-streaming `POST /v1/chat/completions`,
  with extra body fields (`chat_template_kwargs`).
- Replies are JSON in the message text (parsed and schema-validated by the bot),
  not OpenAI tool calling. There is a system prompt worth protecting with a canary.
- The bot makes three kinds of LLM call: owner instructions, plus autonomous
  director and goal decisions that carry state the bot writes itself.
- Today only the owner reaches the LLM. The public page has no LLM: its Q&A answers
  from fixed rules and its buttons are a fixed set of actions. The bot already
  rate-limits public visitors by real IP.

```text
public visitor ─> cobblebob public page ─> public LLM Q&A ─> Aegis ─> LLM
owner ─────────> cobblebob private page ─────── direct ───────────────┘
bot's autonomous calls ───────────────────────── direct ───────────────┘
```

Only public traffic goes through Aegis. Aegis and the LLM listen on loopback only.

## Where Aegis stands (2026-10-24)

**Works today**
- OpenAI-compatible proxy: `POST /v1/chat/completions`, non-streaming
  (cobblebob doesn't stream, so that's enough).
- 6-stage pipeline: rate limit → input classifier → canary → backend → output
  guard (canary leak, PII redaction, harmful keywords) → SQLite audit log.
- Fails closed: classifier errors block, backend errors return 502.
- Trained models: bundled sklearn (3.5 MB) and Phase 3 mDeBERTa (PyTorch + FP32 ONNX,
  1.1 GB each, gitignored). The INT8 ONNX graph (323 MB) is on the HF Hub
  (`billybitcoin/aegis-ml-classifier`), where the HF Space uses it.
- External benchmark: sklearn 91.9% recall / 1.8% FPR; cascade2 FP32 93.5% / 1.2%;
  cascade2 INT8 89.8% / 0.9%.
- HF Space demo, Docker/Compose (sklearn only), CI, tests.

**Added for the cobblebob deployment (M2, 2026-10-24)**
- `GET /v1/models` passthrough; unknown request fields pass through.
- `ENFORCEMENT_MODE=monitor`, `SCAN_ROLES`, `ADMIN_TOKEN`, `CORS_ALLOW_ORIGINS`,
  `AUDIT_RETENTION_DAYS`; JSON-safe PII redaction; audit text limited to scanned roles.
- Torch-free `serve` extra, `deploy/` systemd user unit + install script.

**Still weak**
- Attacks buried in long documents: only 15–24% caught. Every model is weak here.
- No real public traffic reviewed yet (M4 monitor period started 2026-10-24).

---

## Milestones

### M0 · Pin down the target ✅
- [x] Answer the open questions (see [the target](#the-target)).
- [x] Draw the real topology.
- [x] Commit the pending FP32-default changes (tests pass: 95 passed, 18 skipped).
- [x] Pick the serving classifier (see decisions: `cascade2` INT8 on CPU, to verify in M3).

### M1 · Cut scope ✅
Goal: the repo contains one clear product, not a pile of experiments.

- [x] Remove Phase 2 (`hf`, `onnx`, `cascade` classifier types, `training/phase2_hf`,
      the `hf` and `onnx` extras).
- [x] Keep exactly three serving paths: `sklearn`, `cascade2`, `onnx2`. `hf2`
      (PyTorch) serving dropped; training still produces the PyTorch model and
      exports ONNX.
- [x] Keep the INT8 code path: production and the HF Space both rely on it.
- [x] Delete round-2 / targeted experiment code (`training/round2`,
      `training/targeted_acceptance.py`, `prepare_dataset_targeted.py`,
      `targeted_gen.py`). The frozen eval sets stay and `training.eval_suite`
      scores them; BENCHMARKS.md keeps the write-ups and points at commit `a0d4da1`.
      The untracked `docs/prompts/` and `data/round2/base_pool.csv` were moved to
      the local research cache, not deleted.
- [x] Trim the README results section to the current models; link the history.

### M2 · Make Aegis fit cobblebob ✅
Goal: cobblebob's public Q&A path can use Aegis as a drop-in base URL.

- [x] **`GET /v1/models`** passthrough to the backend.
- [x] **Unknown body fields pass through** unchanged (`chat_template_kwargs`), tested.
- [x] **Monitor-only mode**: `ENFORCEMENT_MODE=monitor` records input blocks as
      `would_block` and forwards. The output guard stays enforced in both modes
      (see decisions).
- [x] **Scan scope**: `SCAN_ROLES` (production: `user`). Measured: cascade2 flags
      cobblebob's own public system prompt (0.92 FP32 / 0.34 INT8), so this is required.
- [x] **JSON-safe output guard**: PII inside a JSON object is redacted in its string
      values and the object re-serialised; numbers are never touched.
- [x] **Canary check** against the live model: Qwen did not echo `[SYS_REF:…]`
      (M3 smoke test and the M4 live check).
- [x] **Block response**: Aegis 403 → Blockhead's in-character refusal (`BLOCKED_ANSWER`).
- [x] **Rate limiting**: per-visitor limits stay in Blockhead (unchanged); Aegis's
      60/min is a global cap behind loopback.
- [x] **Admin endpoints**: `ADMIN_TOKEN`; `/audit/logs` disabled without one.
- [x] **CORS**: off unless `CORS_ALLOW_ORIGINS` is set.
- [x] **Compose threshold**: Docker no longer pins 0.70.
- [x] **Privacy**: visitor questions are kept for review, deleted after 30 days
      (`AUDIT_RETENTION_DAYS=30`); the audit text holds only the scanned (user) message.
- [x] Replay test: `tests/test_cobblebob_replay.py`, fixture generated from Blockhead's
      prompt builder.

**Done:** replay test passes; an end-to-end rehearsal on the dev machine (Blockhead's
real client → installed Aegis → fake llama-server) behaved as designed in both modes.

### M3 · Deploy on the bot's host (dark) ✅
Goal: Aegis runs next to the bot and is ready, with no traffic routed to it yet.

- [x] Measure memory: torch-free cascade2 INT8 ~950 MB RSS; ~2.5 ms fast path,
      ~40 ms p50 escalated, on 4 pinned cores of the same CPU model as the host.
- [x] systemd user unit (`deploy/aegis.service`: loopback, restart on failure,
      4 pinned cores, `Nice=5`, `MemoryMax=2G`) and idempotent `deploy/install.sh`,
      rehearsed in a sandbox.
- [x] **(Blockhead)** Separate base URL used only by the public path
      (`model_questions.base_url`); refuses to run if it equals `llm.base_url`.
- [x] **Run the install on the host** (2026-10-24): 926 MB RSS, 4 pinned cores,
      loopback only, monitor mode.
- [x] Smoke test on the host through the live model: benign questions answered in
      ~2 s, an injection flagged `would_block`, the model refused it, no canary echo,
      bot unaffected.
- [x] Audit DB: retention pruning built in; DB lives on local disk in the runtime dir.
- [ ] Simple metrics view (scrape `/metrics` with the admin token, or a dashboard panel).

**Done when:** Aegis is running on the host, the smoke test passes, and the bot shows
no regressions.

### M4 · Public LLM Q&A behind Aegis — live in monitor mode (since 2026-10-24)
Goal: anyone on the web can ask cobblebob free-form questions, safely.

- [x] **(Blockhead)** Model-backed public Q&A (`src/dashboard/public-model-chat.ts`,
      branch `public-llm-qa`): answers only, public facts only, the visitor's question
      is the only user message, falls back to rules-based answers when the model is
      slow or down, 45 s deadline because the model is shared with the bot.
- [x] **(Blockhead)** Public Q&A goes only through Aegis; per-IP limits and size caps
      unchanged. Off by default (`model_questions.enabled`).
- [x] Merge `public-llm-qa`, enable `questions_enabled` + `model_questions`, deploy
      (runbook step 2, 2026-10-24). Live check: a benign question and an injection
      both answered by the model in ~2.5 s; Aegis allowed the first (0.03) and
      flagged the second `would_block` (1.00); the model refused it in character.
- [x] **(Blockhead)** Fix found in the live check: the bot's safety interrupts
      (fired by its own task changes every few seconds) aborted questions waiting
      on the model, so answers fell back to rules. They now cancel actions only.
- [ ] Optional soft launch: share the link with a few friends before announcing it.
- [x] Launch in **monitor mode** (2026-10-24).
- [ ] Review `would_block` rows by hand for 1–2 weeks.
      Record attacks vs false alarms.
- [ ] Tune the threshold on real traffic if needed (record it in decisions).
- [~] **Enforcement guarantees for the public caller** (branch `public-enforcement`,
      2026-10-29, not deployed): `x-aegis-require-enforcement` refuses (503) in monitor
      mode or without a classifier; classifier errors/timeouts block in every mode
      (`guard_unavailable`); `GET /aegis/readiness` (`aegis.readiness/1`); no backend
      redirects; caller disconnect cancels the backend request, and `BACKEND_STREAM`
      makes llama-server actually stop; content-free error logs; `deploy/public-enforce.*`.
      **(Blockhead)** public questions require that readiness, the header and an
      `x-aegis-enforcement: enforce` reply, and hold the model until Aegis shows no open
      backend request. Model Q&A was switched off on 2026-10-29 pending this.
- [ ] Switch to **enforce**; a blocked visitor gets the in-character refusal.
- [ ] Record the first real blocked attacks (redacted) for the write-up.

**Done when:** public Q&A has been live under enforcement for a week with a known,
acceptable false-alarm rate.

### M5 · Portfolio rewrite
Goal: the repo becomes one of the strongest on the GitHub profile.

- [ ] README leads with the real deployment: "guards a public Minecraft bot's LLM",
      the actual topology, and live numbers (requests, blocks, false-alarm rate).
- [ ] One or two real, redacted blocked attempts as examples.
- [ ] Short "design decisions and trade-offs" section (fail-closed, cascade,
      monitor-then-enforce, INT8 vs FP32 on a small home server).
- [ ] Research history stays in BENCHMARKS.md, linked, not in the README.
- [ ] Optional: a write-up ("I let the internet talk to my Minecraft bot").
- [ ] Refresh the HF Space text so it points to the production story.

**Done when:** someone landing on the repo sees a deployed, measured security service first.

### M6 · Public free-form commands (stretch)
Goal: visitors can give cobblebob tasks like "collect 8 oak logs".

Only after M4 has run under enforcement long enough to trust it.

- [ ] Threat model the action surface: which tools are safe for the public,
      per-tool limits, never anything that touches the host beyond the game.
- [ ] **(Blockhead)** Public command path with an allowlist of tools, separate from
      owner commands.
- [ ] Output-side checks on the chosen action (the JSON decision), not just the text.
- [ ] Same monitor → enforce rollout as M4.

**Done when:** public commands run under enforcement with no griefing incidents.

---

## Decisions

Record new decisions in this format: date, decision, why.

- **2026-10-24 · Goal: public Q&A first, then public commands.** Today only the owner
  reaches the LLM. Opening it to the public is the reason Aegis exists in this setup.
- **2026-10-24 · Aegis runs on the bot's host, loopback only.** Same machine as cobblebob and
  llama-server; nothing new is exposed to the network.
- **2026-10-24 · Only public traffic goes through Aegis.** The owner's own prompts
  are never filtered (owner's call). Autonomous director and goal calls carry the
  bot's own state, not untrusted text, so scanning them would risk false positives
  pausing the bot for no security gain. Consequence: there's no owner traffic to
  trial Aegis on, so M3 deploys it dark and M4 starts in monitor mode.
- **2026-10-24 · Production classifier: `cascade2` with INT8 on CPU (to verify in M3).**
  Supersedes "cascade2 FP32" from earlier the same day. The host's free RAM and VRAM
  can't comfortably fit the 1.1 GB FP32 graph next to the LLM. INT8
  (323 MB) scores 89.8% recall / 0.9% FPR on the external benchmark, and the cascade
  sends only ~24% of prompts to it. Fallback: sklearn only.
- **2026-10-24 · INT8 code stays.** Both production and the HF Space rely on it; production
  downloads the graph from the Hub repo. The hf2 training checkpoints and the local
  INT8 file were deleted from the dev machine to free ~4.3 GB.
- **2026-10-24 · No streaming work.** cobblebob sends `stream: false` and waits for the full reply.
- **2026-10-24 · Classifier research is paused.** Rounds 1–2 produced no shippable
  model. Resume only with real-traffic evidence from M4 (parking lot).
- **2026-10-24 · Monitor mode relaxes only the input classifier.** That's where
  false positives come from. Canary-leak blocking and PII redaction stay on in
  monitor mode: they almost never misfire, and what they catch (a leaked system
  prompt, PII) would go straight to the public.
- **2026-10-24 · Public answers fall back to rules, never fail.** If the model is
  slow (it queues behind the bot's own decisions), down or blocked, the visitor
  still gets the existing rules-based reply or the in-character refusal.
- **2026-10-24 · Single worker.** The canary store is in-memory, and llama-server
  serializes requests anyway, so the LLM is the bottleneck, not Aegis.

- **2026-10-29 · Public callers can require enforcement per request.** Monitor mode
  stays useful for other callers, but cobblebob's public Q&A must never be forwarded
  unguarded, so it sends a header that turns monitor mode or an unloaded classifier
  into a 503 before any classification or backend work. Readiness is versioned so
  a generic 200 (llama.cpp's own endpoints) can't pass for it.
- **2026-10-29 · Classifier failure blocks in every mode.** Monitor mode used to forward
  a request whose classification failed; an unscanned request is now never forwarded.
- **2026-10-29 · Stream from llama-server to make cancellation real.** Measured on
  llama.cpp b11425: closing a non-streamed request's connection usually leaves the
  task running to completion; a streamed request is cancelled at its next token.
  `BACKEND_STREAM` keeps the caller-facing reply non-streaming.
- **2026-10-29 · Audit keeps no visitor text for the public deployment.**
  `REDACT_PROMPTS_IN_LOGS=true`, 7-day retention; aiosqlite's DEBUG logger (which
  echoes SQL parameters) is silenced, and errors log exception types only.

## Parking lot

Ideas that are not part of the current path. Pull one in only with a reason.

- Better detection of attacks buried in long documents (the weakest result).
- Retrain on real cobblebob traffic labels from M4 (privacy review first).
- FP32 on the host's GPU if VRAM frees up (check onnxruntime CUDA support for its GPU).
- Redis canary store / multiple workers.
- OpenAI tool-calling support (`tool_calls`, `tool_call_id` are dropped by
  `ChatMessage` today); not needed while cobblebob uses JSON in text.
- Streaming (SSE) proxying.
- Semantic output checks beyond keywords (e.g. a small judge model).
- Per-route or per-caller policies (different thresholds per cobblebob feature).
- Packaging as a reusable Compose bundle for others.

## Changelog

Newest first. One line per shipped item.

- **2026-10-29** Enforcement guarantees for the public caller built and validated on
  an isolated stack (branch `public-enforcement`); see Blockhead
  `docs/public-model-answer-enablement.md`. Not deployed: production is still
  `ec36ec9` in monitor mode, and cobblebob's model Q&A is off.

- **2026-10-24** M3 done: Aegis installed on the bot's host (dark), smoke test
  passed through the live model with no canary echo. M4 launched in monitor mode:
  public visitors' questions are answered by the LLM through Aegis. Fixed in
  Blockhead during the live check: safety interrupts no longer cut off questions.

- **2026-10-24** M2 done (Aegis: `/v1/models`, monitor mode, `SCAN_ROLES`, admin
  token, CORS, retention, JSON-safe redaction, `serve` extra, replay test). M3
  prepared (`deploy/`, rehearsed). M4 Blockhead code built on branch `public-llm-qa`.
  Measured cobblebob traffic: cascade2 INT8 allowed all 15 benign questions and
  blocked all 6 attacks; sklearn false-flagged "tell me a joke" and "what are your rules?".
- **2026-10-24** M1 done: removed Phase 2, PyTorch serving and the round-2/targeted
  experiment code; README results trimmed to sklearn / cascade2 / onnx2.
- **2026-10-24** Owner prompts stay unfiltered: only public traffic goes through
  Aegis; M3 is now a dark deploy plus smoke test.
- **2026-10-24** M0 done: target, topology and host sizing recorded; production
  classifier changed to cascade2 INT8; roadmap restructured around public Q&A (M4)
  and public commands (M6). Committed the FP32-default ONNX changes.
- **2026-10-24** Deleted hf2 training checkpoints (4 GB), the local INT8 ONNX graph
  (still on the Hub) and the round-2 leftover models; restored the round-2 baseline
  copy that `training/round2/diagnose.py` reads. Wrote this roadmap.
