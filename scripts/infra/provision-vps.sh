#!/usr/bin/env bash
set -euo pipefail

# MES Scalping Bot — VPS Provisioning Script
# Target: Ubuntu 22.04/24.04 LTS on Vultr Chicago
# Run as root: sudo bash provision-vps.sh
# Idempotent: safe to run multiple times.

BOT_USER="botuser"
BOT_DIR="/opt/mes-bot"
SERVICE_NAME="mes-bot"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() { echo -e "\n${GREEN}[STEP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ── Preflight ───────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    fail "This script must be run as root. Use: sudo bash provision-vps.sh"
fi

# ── 1. System packages ─────────────────────────────────────────────────────
step "Updating apt and installing system packages"
apt-get update -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv python3-dev \
    git htop tmux curl wget jq \
    build-essential libffi-dev libssl-dev \
    > /dev/null 2>&1

PYTHON_VERSION=$(python3 --version 2>&1)
echo "  Installed: $PYTHON_VERSION"

# ── 2. Create bot user ─────────────────────────────────────────────────────
step "Creating bot user: $BOT_USER"
if id "$BOT_USER" &>/dev/null; then
    echo "  User $BOT_USER already exists, skipping"
else
    useradd -m -s /bin/bash "$BOT_USER"
    usermod -aG sudo "$BOT_USER"
    echo "  Created user $BOT_USER with sudo access"
fi

# ── 3. Bot directory ───────────────────────────────────────────────────────
step "Creating bot directory: $BOT_DIR"
mkdir -p "$BOT_DIR"
chown "$BOT_USER:$BOT_USER" "$BOT_DIR"
echo "  $BOT_DIR owned by $BOT_USER"

# ── 4. Python virtual environment + uvloop ──────────────────────────────────
step "Setting up Python virtual environment"
VENV_DIR="$BOT_DIR/venv"
if [[ -d "$VENV_DIR" ]]; then
    echo "  Virtual environment already exists at $VENV_DIR"
else
    sudo -u "$BOT_USER" python3 -m venv "$VENV_DIR"
    echo "  Created virtual environment at $VENV_DIR"
fi

step "Installing uvloop"
sudo -u "$BOT_USER" "$VENV_DIR/bin/pip" install --quiet --upgrade pip
sudo -u "$BOT_USER" "$VENV_DIR/bin/pip" install --quiet uvloop

UVLOOP_CHECK=$("$VENV_DIR/bin/python" -c "import uvloop; print(f'uvloop {uvloop.__version__}')" 2>&1) || true
if [[ "$UVLOOP_CHECK" == uvloop* ]]; then
    echo "  $UVLOOP_CHECK installed successfully"
else
    warn "uvloop installation may have failed: $UVLOOP_CHECK"
fi

# ── 5. UFW Firewall ────────────────────────────────────────────────────────
step "Configuring UFW firewall"
apt-get install -y -qq ufw > /dev/null 2>&1
ufw --force reset > /dev/null 2>&1
ufw default deny incoming > /dev/null 2>&1
ufw default allow outgoing > /dev/null 2>&1
ufw allow ssh > /dev/null 2>&1
ufw --force enable > /dev/null 2>&1
echo "  UFW enabled: SSH allowed, all other inbound denied"

# ── 6. fail2ban ─────────────────────────────────────────────────────────────
step "Installing fail2ban"
apt-get install -y -qq fail2ban > /dev/null 2>&1
systemctl enable fail2ban > /dev/null 2>&1
systemctl start fail2ban > /dev/null 2>&1
echo "  fail2ban active for SSH brute-force protection"

# ── 7. systemd service ─────────────────────────────────────────────────────
step "Creating systemd service: $SERVICE_NAME"
cat > "/etc/systemd/system/${SERVICE_NAME}.service" << 'SYSTEMD'
[Unit]
Description=MES Scalping Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=botuser
Group=botuser
WorkingDirectory=/opt/mes-bot
ExecStart=/opt/mes-bot/venv/bin/python -m src.main
Restart=on-failure
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5

# Environment file for secrets (Tradovate credentials, etc.)
EnvironmentFile=-/opt/mes-bot/.env

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mes-bot

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/opt/mes-bot
PrivateTmp=true

[Install]
WantedBy=multi-user.target
SYSTEMD

systemctl daemon-reload
echo "  Service template created (not started — no bot code yet)"

# ── 8. Latency tests ───────────────────────────────────────────────────────
step "Running latency tests"
LATENCY_FILE="/tmp/latency-test.txt"
echo "=== MES Bot Latency Test — $(date -u) ===" > "$LATENCY_FILE"

echo "" >> "$LATENCY_FILE"
echo "--- Ping to CME Globex (198.105.251.100) ---" >> "$LATENCY_FILE"
if ping -c 10 -q 198.105.251.100 >> "$LATENCY_FILE" 2>&1; then
    CME_LATENCY=$(tail -1 "$LATENCY_FILE" | grep -oP '[\d.]+/[\d.]+/' | head -1 | cut -d'/' -f2)
    echo "  CME Globex avg ping: ${CME_LATENCY:-unknown}ms"
else
    warn "CME Globex ping failed (ICMP may be blocked)"
    echo "  CME ping: ICMP blocked or unreachable"
fi

echo "" >> "$LATENCY_FILE"
echo "--- Tradovate API latency ---" >> "$LATENCY_FILE"
TRADO_START=$(date +%s%N)
TRADO_HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "https://live.tradovateapi.com/v1/auth/accesstokenrequest" 2>/dev/null || echo "failed")
TRADO_END=$(date +%s%N)
TRADO_MS=$(( (TRADO_END - TRADO_START) / 1000000 ))
echo "Tradovate API response: HTTP $TRADO_HTTP in ${TRADO_MS}ms" >> "$LATENCY_FILE"
echo "  Tradovate API round-trip: ${TRADO_MS}ms (HTTP $TRADO_HTTP)"

echo "" >> "$LATENCY_FILE"
echo "--- Tradovate Demo API latency ---" >> "$LATENCY_FILE"
DEMO_START=$(date +%s%N)
DEMO_HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "https://demo.tradovateapi.com/v1/auth/accesstokenrequest" 2>/dev/null || echo "failed")
DEMO_END=$(date +%s%N)
DEMO_MS=$(( (DEMO_END - DEMO_START) / 1000000 ))
echo "Tradovate Demo API response: HTTP $DEMO_HTTP in ${DEMO_MS}ms" >> "$LATENCY_FILE"
echo "  Tradovate Demo API round-trip: ${DEMO_MS}ms (HTTP $DEMO_HTTP)"

echo "  Full results saved to $LATENCY_FILE"

# ── 9. Summary ──────────────────────────────────────────────────────────────
step "Provisioning complete"
echo ""
echo "  ┌─────────────────────────────────────────────┐"
echo "  │         MES Bot VPS — Summary               │"
echo "  ├─────────────────────────────────────────────┤"
echo "  │ Python:    $PYTHON_VERSION"
echo "  │ uvloop:    ${UVLOOP_CHECK:-not installed}"
echo "  │ UFW:       $(ufw status | head -1)"
echo "  │ fail2ban:  $(systemctl is-active fail2ban)"
echo "  │ Bot dir:   $BOT_DIR"
echo "  │ Bot user:  $BOT_USER"
echo "  │ Service:   $SERVICE_NAME (inactive — no code yet)"
echo "  │ CME ping:  ${CME_LATENCY:-N/A}ms"
echo "  │ Tradovate: ${TRADO_MS}ms"
echo "  └─────────────────────────────────────────────┘"
echo ""
echo "  Next steps:"
echo "  1. Copy your SSH public key: ssh-copy-id $BOT_USER@<this-ip>"
echo "  2. Disable password auth in /etc/ssh/sshd_config"
echo "  3. Clone your bot repo into $BOT_DIR"
echo "  4. Create $BOT_DIR/.env with Tradovate credentials"
