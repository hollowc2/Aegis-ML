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

**Gaps for the cobblebob deployment**
- **No `GET /v1/models`.** cobblebob calls it first, so it would fail immediately.
- `/audit/logs` and `/metrics` have no auth (less urgent while Aegis is on loopback).
- No monitor-only mode: no way to measure false positives on real traffic before blocking.
- Untested on cobblebob's traffic: canary text in the system prompt, and PII
  redaction or keyword filtering touching JSON replies (numbers or coordinates
  matching the phone or IPv4 patterns would corrupt the JSON).
- Per-IP rate limiting sees every request as loopback behind the bot.
- Attacks buried in long documents: only 15–24% caught. Every model is weak here.
- The Docker image installs no onnxruntime or tokenizer, so it can't run `cascade2`.

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

### M2 · Make Aegis fit cobblebob
Goal: cobblebob's public Q&A path can use Aegis as a drop-in base URL.

- [ ] **`GET /v1/models`** passthrough to the backend (blocker).
- [ ] **Unknown body fields pass through** unchanged (`chat_template_kwargs`); add a test.
- [ ] **Monitor-only mode**: `ENFORCEMENT_MODE=monitor|enforce`. Monitor logs the
      verdict but forwards anyway. Required for M4.
- [ ] **Scan scope**: classify only untrusted text (user messages), not cobblebob's
      own system prompt and state snapshots, or make it configurable per role.
- [ ] **JSON-safe output guard**: run the output guard on real cobblebob replies;
      make sure PII redaction can't corrupt JSON (skip or narrow numeric patterns
      for JSON replies). Keep the canary check.
- [ ] **Canary check**: confirm the bot's model doesn't echo or trip on `[SYS_REF:…]`
      in cobblebob's system prompt.
- [ ] **Block response cobblebob can handle**: a 403 body Blockhead can turn into a
      friendly reply (403 isn't retried by Blockhead's client, which is what we want).
- [ ] **Rate limiting**: per-visitor limits stay in Blockhead; set Aegis's limit as a
      global safety cap, or have Blockhead forward the visitor IP in a trusted header.
- [ ] **Admin endpoints**: bearer token on `/audit/logs` and `/metrics`, locked by default.
- [ ] **CORS**: the app allows every origin ("permissive for demo"); restrict it.
- [ ] **Compose threshold**: `docker-compose.yml` hard-codes `CONFIDENCE_THRESHOLD=0.70`,
      overriding each model's tuned threshold; leave it unset by default.
- [ ] **Privacy**: decide what visitor text is kept (`REDACT_PROMPTS_IN_LOGS`,
      retention period, log rotation).
- [ ] Replay test: a fixture of cobblebob-shaped requests and responses (the
      public Q&A prompt plus benign and hostile visitor questions) runs through
      Aegis in CI.

**Done when:** the replay test passes, and a local cobblebob public Q&A call
behaves the same through Aegis as it does straight to llama-server.

### M3 · Deploy on the bot's host (dark)
Goal: Aegis runs next to the bot and is ready, with no traffic routed to it yet.

- [ ] Measure memory: Aegis with `cascade2` INT8 on CPU must fit in the host's
      memory headroom (see `DEPLOYMENT.local.md`). Pin it to a few cores (`taskset`); don't starve llama-server or Minecraft.
- [ ] Run Aegis as a systemd user unit on the host, loopback only, restart on failure.
- [ ] **(Blockhead)** Add a separate LLM base-URL setting used only by the public
      path. Owner calls and autonomous director/goal calls keep going straight to
      llama-server, unfiltered. Rollback is one config line.
- [ ] Smoke test on the host: replay the M2 fixture against the live llama-server
      through Aegis; check latency and that the bot keeps running normally.
- [ ] Back up the audit DB; set up log rotation.
- [ ] Simple metrics view (scrape `/metrics`, or a panel on the Blockhead dashboard).

**Done when:** Aegis is running on the host, the smoke test passes, and the bot shows
no regressions.

### M4 · Public LLM Q&A behind Aegis
Goal: anyone on the web can ask cobblebob free-form questions, safely.

- [ ] **(Blockhead)** Build model-backed public Q&A: answers only, no tools or
      actions, redacted state only (reuse the public facts and the owner
      viewer-chat prompt's "question is data" rules).
- [ ] **(Blockhead)** Send public Q&A only through Aegis; keep the existing per-IP
      rate limits and size caps.
- [ ] Optional soft launch: share the link with a few friends before announcing it.
- [ ] Launch in **monitor mode**; review flagged questions by hand for 1–2 weeks.
      Record attacks vs false alarms.
- [ ] Tune the threshold on real traffic if needed (record it in decisions).
- [ ] Switch to **enforce**; a blocked visitor gets a friendly in-character reply.
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
- **2026-10-24 · Single worker.** The canary store is in-memory, and llama-server
  serializes requests anyway, so the LLM is the bottleneck, not Aegis.

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
