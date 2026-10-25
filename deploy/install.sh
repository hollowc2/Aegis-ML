#!/usr/bin/env bash
# Install or update Aegis-ML as a systemd user service on the host that runs
# the LLM. Idempotent; never starts or restarts the service unless --start.
#
#   deploy/install.sh --source /path/to/Aegis-ML --ref cobblebob-prod [--start]
#
# Layout (local disk, not NFS):
#   ~/.local/share/aegis-runtime        code checkout + .venv + models + logs
#   ~/.config/aegis/aegis.env           settings (created once, chmod 600)
#   ~/.config/systemd/user/aegis.service
set -euo pipefail

SOURCE=""
REF="main"
START=0
AEGIS_HOME="${AEGIS_HOME:-$HOME/.local/share/aegis-runtime}"
CONFIG_DIR="$HOME/.config/aegis"
UNIT_DIR="$HOME/.config/systemd/user"
MODEL_REPO="${AEGIS_MODEL_REPO:-billybitcoin/aegis-ml-classifier}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --start) START=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$SOURCE" ]] || { echo "--source is required (git URL or path)" >&2; exit 2; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

echo "==> code: $SOURCE @ $REF -> $AEGIS_HOME"
if [[ -d "$AEGIS_HOME/.git" ]]; then
  git -C "$AEGIS_HOME" fetch --quiet "$SOURCE" "$REF"
  git -C "$AEGIS_HOME" checkout --quiet --detach FETCH_HEAD
else
  git clone --quiet "$SOURCE" "$AEGIS_HOME"
  git -C "$AEGIS_HOME" checkout --quiet --detach "origin/$REF" 2>/dev/null \
    || git -C "$AEGIS_HOME" checkout --quiet --detach "$REF"
fi
echo "    at $(git -C "$AEGIS_HOME" log --oneline -1)"

echo "==> venv (serve extra: onnxruntime + tokenizer, no PyTorch)"
(cd "$AEGIS_HOME" && UV_PROJECT_ENVIRONMENT="$AEGIS_HOME/.venv" uv sync --quiet --locked --no-dev --extra serve)

ONNX_DIR="$AEGIS_HOME/models/hf2_classifier_onnx"
if [[ -f "$ONNX_DIR/model_int8.onnx" && -f "$ONNX_DIR/config.json" ]]; then
  echo "==> INT8 model already present"
else
  echo "==> INT8 model from $MODEL_REPO (FP32 graph skipped)"
  "$AEGIS_HOME/.venv/bin/python" - "$MODEL_REPO" "$ONNX_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], repo_type="model",
                  ignore_patterns=["model.onnx", "model.onnx.data"])
PY
fi
mkdir -p "$AEGIS_HOME/logs"

mkdir -p "$CONFIG_DIR"
if [[ -f "$CONFIG_DIR/aegis.env" ]]; then
  echo "==> keeping existing $CONFIG_DIR/aegis.env"
else
  echo "==> writing $CONFIG_DIR/aegis.env (monitor mode, new admin token)"
  token="$("$AEGIS_HOME/.venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))')"
  sed "s|^ADMIN_TOKEN=.*|ADMIN_TOKEN=$token|" "$AEGIS_HOME/deploy/aegis.env.example" > "$CONFIG_DIR/aegis.env"
  chmod 600 "$CONFIG_DIR/aegis.env"
fi

mkdir -p "$UNIT_DIR"
install -m 644 "$AEGIS_HOME/deploy/aegis.service" "$UNIT_DIR/aegis.service"
systemctl --user daemon-reload
echo "==> unit installed: $UNIT_DIR/aegis.service"

if [[ "$START" == 1 ]]; then
  systemctl --user enable --now aegis.service
  systemctl --user restart aegis.service
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$(grep -E '^PORT=' "$CONFIG_DIR/aegis.env" | cut -d= -f2)/health"; then echo; exit 0; fi
    sleep 1
  done
  echo "Aegis did not report healthy within 60 s; see: journalctl --user -u aegis -n 50" >&2
  exit 1
else
  echo "Not started. Start with: systemctl --user enable --now aegis.service"
fi
