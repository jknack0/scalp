#!/usr/bin/env bash
set -euo pipefail

# Set up weekly auto-retrain cron for botuser
# Run as root on the VPS:  bash /opt/mes-bot/scripts/infra/setup-retrain-cron.sh

BOT_DIR="/opt/mes-bot"
UV_PATH="/home/botuser/.local/bin/uv"

# 1. Allow botuser to restart mes-bot without password
echo "botuser ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart mes-bot" > /etc/sudoers.d/mes-bot-retrain
chmod 440 /etc/sudoers.d/mes-bot-retrain
echo "[OK] botuser can now restart mes-bot without password"

# 2. Install cron job for botuser
# Runs every Saturday at 06:00 UTC (1 AM ET, market closed)
CRON_LINE="0 6 * * 6 cd $BOT_DIR && $UV_PATH run python scripts/train/auto_retrain.py >> logs/retrain.log 2>&1"

# Check if cron already exists
if sudo -u botuser crontab -l 2>/dev/null | grep -q "auto_retrain"; then
    echo "[SKIP] Retrain cron already exists"
else
    (sudo -u botuser crontab -l 2>/dev/null || true; echo "$CRON_LINE") | sudo -u botuser crontab -
    echo "[OK] Weekly retrain cron installed for botuser"
fi

echo ""
echo "Retrain schedule: Every Saturday 06:00 UTC (01:00 AM ET)"
echo "Logs: $BOT_DIR/logs/retrain.log"
echo ""
echo "Manual run:"
echo "  cd $BOT_DIR && $UV_PATH run python scripts/train/auto_retrain.py"
echo ""
echo "Dry run (validate only):"
echo "  cd $BOT_DIR && $UV_PATH run python scripts/train/auto_retrain.py --dry-run"
