# Threat model

## Security objective

Aegis sits between an application and an OpenAI-compatible language-model
backend. Its objective is to reduce the chance that untrusted prompt content
changes model behavior, extracts hidden instructions, or returns obvious
sensitive data.

## Trust boundaries

- Client messages and retrieved content are untrusted.
- The Aegis service, its configuration, and model artifacts are trusted.
- The backend LLM is treated as fallible and potentially influenced by hostile
  input.
- Model output is untrusted until it passes the output guardrail.
- Operators are responsible for access control, transport security, secrets,
  and network isolation around the service.

## Controls

1. Per-IP rate limiting reduces automated abuse.
2. The input classifier blocks prompts at or above the configured threshold.
3. A per-request canary is injected into the system context.
4. Canary disclosure in model output causes the response to be blocked.
5. PII patterns are redacted and a small harmful-output ruleset can block.
6. Decisions and latency are written to an audit log and exported as metrics.
7. Classifier exceptions fail secure by blocking the input.

## Known limitations

- No classifier detects every direct, indirect, encoded, multilingual, or
  multi-turn injection.
- Canary detection proves only that the exact token leaked. Its absence does not
  prove that the model followed its intended instructions.
- Regex-based PII detection is incomplete and can both miss and over-redact.
- The harmful-output filter is deliberately narrow and is not a general content
  safety classifier.
- The canary registry is in memory and is not shared across workers or replicas.
- SQLite is suitable for a demonstration or single-node deployment, not a
  high-volume distributed audit system.
- `/audit/logs` has no built-in authentication and must not be exposed publicly
  without an authenticated gateway.
- Model artifacts loaded with Python serialization formats such as `joblib`
  must come from a trusted source.
- The proxy currently focuses on non-streaming chat completions; complete
  OpenAI API compatibility is not claimed.

## Deployment guidance

- Run one worker unless the canary store is replaced with a shared store.
- Put the service behind TLS and authentication.
- Restrict the backend model so clients cannot bypass Aegis.
- Enable prompt redaction when audit content may contain personal data.
- Pin and verify model artifacts.
- Monitor block rates, error rates, latency, and classifier drift.
- Re-evaluate against representative application traffic before changing the
  threshold.

## Reporting a vulnerability

Do not include sensitive exploit details in a public issue. Use GitHub's private
security-advisory flow for this repository when available, or contact the
maintainer through the account listed on the repository.
