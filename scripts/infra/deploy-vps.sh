#!/usr/bin/env bash
set -euo pipefail

# MES Scalping Bot — VPS Deployment Script
# Run as botuser on the VPS:
#   bash /tmp/deploy-vps.sh
#
# Prerequisites:
#   - VPS provisioned (provision-vps.sh already run)
#   - SSH key added to GitHub for botuser
#   - .env file ready with Tradovate demo credentials

BOT_DIR="/opt/mes-bot"
REPO_URL="https://github.com/jknack0/scalp.git"
SERVICE_NAME="mes-bot"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() { echo -e "\n${GREEN}[STEP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ── 1. Install uv ────────────────────────────────────────────────────────
step "Installing uv package manager"
if command -v uv &>/dev/null; then
    echo "  uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "  Installed: $(uv --version)"
fi

# ── 2. Clone or update repo ──────────────────────────────────────────────
step "Setting up repository at $BOT_DIR"
if [[ -d "$BOT_DIR/.git" ]]; then
    echo "  Repo exists, pulling latest..."
    cd "$BOT_DIR"
    git fetch origin
    git reset --hard origin/main
    echo "  Updated to latest main"
else
    echo "  Cloning fresh..."
    # If directory exists but isn't a git repo, clear it
    if [[ -d "$BOT_DIR" ]]; then
        rm -rf "$BOT_DIR"/*
        rm -rf "$BOT_DIR"/.[^.]*
    fi
    git clone "$REPO_URL" "$BOT_DIR"
    cd "$BOT_DIR"
    echo "  Cloned to $BOT_DIR"
fi

# ── 3. Install Python dependencies ───────────────────────────────────────
step "Installing Python dependencies with uv"
cd "$BOT_DIR"
uv sync --no-dev
echo "  Dependencies installed"

# ── 4. Create directories ────────────────────────────────────────────────
step "Creating runtime directories"
mkdir -p "$BOT_DIR/logs"
echo "  logs/ created"

# ── 5. Check .env ────────────────────────────────────────────────────────
step "Checking .env credentials"
if [[ -f "$BOT_DIR/.env" ]]; then
    echo "  .env found"
    if grep -q "TRADOVATE_USERNAME=" "$BOT_DIR/.env" && \
       ! grep -q "TRADOVATE_USERNAME=$" "$BOT_DIR/.env"; then
        echo "  Tradovate credentials appear configured"
    else
        warn ".env exists but TRADOVATE_USERNAME may be empty"
    fi
else
    warn ".env not found! Copy from your local machine:"
    echo "    scp .env botuser@$(hostname -I | awk '{print $1}'):$BOT_DIR/.env"
    echo "    chmod 600 $BOT_DIR/.env"
fi

# ── 6. Update systemd service ────────────────────────────────────────────
step "Updating systemd service for uv"
UV_PATH=$(which uv 2>/dev/null || echo "$HOME/.local/bin/uv")
sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null << EOF
[Unit]
Description=MES Scalping Bot (Paper Trading)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=botuser
Group=botuser
WorkingDirectory=$BOT_DIR
ExecStart=$UV_PATH run python main.py
Restart=on-failure
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5

# Load .env for credentials
EnvironmentFile=-$BOT_DIR/.env

# PATH for uv
Environment="PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mes-bot

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$BOT_DIR
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo "  Service updated: $SERVICE_NAME"

# ── 7. Smoke test ────────────────────────────────────────────────────────
step "Running smoke test"
cd "$BOT_DIR"
$UV_PATH run python -c "
from src.core.config import BotConfig
from src.strategies.orb_strategy import ORBConfig, ORBStrategy
from src.strategies.vwap_strategy import VWAPConfig, VWAPStrategy
from src.filters.spread_monitor import SpreadConfig, SpreadMonitor
print('  Config loads: OK')
print('  ORB strategy: OK')
print('  VWAP strategy: OK')
print('  Spread filter: OK')
"

# ── 8. Summary ────────────────────────────────────────────────────────────
step "Deployment complete!"
echo ""
echo "  ┌─────────────────────────────────────────────┐"
echo "  │      MES Bot — Deployment Summary           │"
echo "  ├─────────────────────────────────────────────┤"
echo "  │ Mode:       PAPER TRADING (demo API)        │"
echo "  │ Strategies: ORB + VWAP                      │"
echo "  │ Filters:    Spread z=2.0 (ORB only)         │"
echo "  │ Symbol:     MESH6                           │"
echo "  │ Bar size:   5s                              │"
echo "  └─────────────────────────────────────────────┘"
echo ""
echo "  Commands:"
echo "    sudo systemctl start mes-bot     # Start the bot"
echo "    sudo systemctl enable mes-bot    # Auto-start on boot"
echo "    journalctl -u mes-bot -f         # Tail logs"
echo "    sudo systemctl stop mes-bot      # Stop the bot"
echo ""
if [[ ! -f "$BOT_DIR/.env" ]]; then
    echo "  ⚠ BEFORE STARTING: Copy your .env file!"
    echo "    scp .env botuser@VPS_IP:$BOT_DIR/.env"
    echo "    ssh botuser@VPS_IP 'chmod 600 $BOT_DIR/.env'"
    echo ""
fi
