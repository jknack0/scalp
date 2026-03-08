# VPS Setup Guide — MES Scalping Bot

> Phase 1, Task 2 | Last updated: 2026-03-03

---

## Why a VPS?

The VPS is for **reliability**, not latency. Your strategies target 2–30 minute holds with 8+ tick profit targets — the difference between 5ms and 50ms execution is a fraction of a tick. What kills you is your home internet dropping during a live position with no stop-loss server-side.

### Tradovate Latency Reality

The original roadmap assumed Rithmic, which connects directly to CME in Aurora, IL (sub-1ms from a Chicago VPS). **Tradovate is different** — its API servers are cloud-hosted (Google Cloud), so the latency path is:

```
Your VPS  →  Tradovate API (cloud)  →  CME Globex (Aurora, IL)
```

A Chicago VPS gives you geographic proximity to CME but **not** to Tradovate's API servers. Real-world Tradovate API round-trips are typically 20–80ms regardless of your VPS location. This is fine for your strategy timescales.

**Bottom line:** Pick a VPS based on cost and reliability, not latency.

---

## Provider Recommendation

### Vultr Chicago — $6-12/mo (Recommended)

| Plan | vCPU | RAM | Storage | Bandwidth | Price |
|---|---|---|---|---|---|
| Cloud Compute (vc2-1c-1gb) | 1 | 1 GB | 25 GB SSD | 1 TB | **$6/mo** |
| Cloud Compute (vc2-1c-2gb) | 1 | 2 GB | 50 GB SSD | 2 TB | **$12/mo** |
| Cloud Compute (vc2-2c-4gb) | 2 | 4 GB | 80 GB SSD | 3 TB | **$24/mo** |

**Start with the $6/mo plan.** A Python asyncio bot uses minimal resources. Upgrade if you add TimescaleDB on the same box later.

Sign up: [vultr.com](https://www.vultr.com/) → Deploy New Server → **Chicago** datacenter → Ubuntu 22.04 LTS.

### Other Options (Not Recommended for Now)

| Provider | Price | When to Consider |
|---|---|---|
| QuantVPS Chicago | $60-100/mo | If you switch to Rithmic later (sub-1ms to CME). Overkill for Tradovate. |
| Contabo | $5-10/mo | Budget option, but inconsistent network quality. |
| Hetzner | $4-8/mo | No Chicago datacenter (US East only). |

---

## Setup Steps

### 1. Deploy the Server

1. Go to [vultr.com](https://www.vultr.com/) and create an account
2. Click **Deploy New Server**
3. Select:
   - Type: **Cloud Compute — Shared CPU**
   - Location: **Chicago**
   - Image: **Ubuntu 22.04 LTS** (or 24.04 LTS)
   - Plan: **$6/mo** (1 vCPU, 1 GB RAM)
4. Under **SSH Keys**, add your public key (see below if you don't have one)
5. Click **Deploy Now**
6. Wait ~60 seconds for the server to spin up
7. Copy the IP address from the dashboard

### 2. Generate an SSH Key (if you don't have one)

On your local machine (Windows PowerShell or Git Bash):

```bash
ssh-keygen -t ed25519 -C "mes-bot-vps"
```

Press Enter for defaults. Your public key is at `~/.ssh/id_ed25519.pub`. Copy its contents and paste into Vultr's SSH Key field.

### 3. SSH into the Server

```bash
ssh root@66.42.124.72
```

### 4. Run the Provisioning Script

Upload and run the script:

```bash
# From your local machine, upload the script
scp scripts/infra/provision-vps.sh root@66.42.124.72:/tmp/

# SSH in and run it
ssh root@66.42.124.72
bash /tmp/provision-vps.sh
```

The script will:
- Install Python 3.11+, uvloop, git, tmux, htop
- Create a `botuser` account with sudo access
- Configure UFW firewall (SSH only)
- Install fail2ban for brute-force protection
- Create `/opt/mes-bot` directory
- Set up a systemd service template
- Run latency tests to CME and Tradovate API
- Print a summary

### 5. Disable Password Authentication

After confirming SSH key login works:

```bash
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart sshd
```

### 6. Verify Latency

Check the latency test results:

```bash
cat /tmp/latency-test.txt
```

Expected results:
- **CME Globex ping**: 1-5ms from Chicago (may show N/A if ICMP is blocked)
- **Tradovate API**: 20-80ms (this is normal — cloud-hosted servers)

---

## Systemd Service Management

Once the bot code is deployed, manage it with:

```bash
# Start the bot
sudo systemctl start mes-bot

# Stop the bot
sudo systemctl stop mes-bot

# Restart the bot
sudo systemctl restart mes-bot

# Check status
sudo systemctl status mes-bot

# View logs (live tail)
journalctl -u mes-bot -f

# View last 100 log lines
journalctl -u mes-bot -n 100

# Enable auto-start on boot
sudo systemctl enable mes-bot
```

The service is configured to:
- Restart automatically on crash (after 10s delay)
- Max 5 restarts within 5 minutes (prevents crash loops)
- Load environment variables from `/opt/mes-bot/.env`
- Log to systemd journal

---

## Environment File

Create `/opt/mes-bot/.env` for Tradovate credentials:

```bash
sudo -u botuser nano /opt/mes-bot/.env
```

```
TRADOVATE_USERNAME=your_username
TRADOVATE_PASSWORD=your_password
TRADOVATE_APP_ID=your_app_id
TRADOVATE_APP_VERSION=1.0.0
TRADOVATE_CID=your_cid
TRADOVATE_SECRET=your_secret
TRADOVATE_DEMO=true
```

Secure it:

```bash
chmod 600 /opt/mes-bot/.env
```

---

## Troubleshooting

### Tradovate API latency > 200ms
- This is a Tradovate server issue, not your VPS. Check [Tradovate status](https://community.tradovate.com/) for outages.
- Try the demo endpoint as a comparison: `curl -w "%{time_total}" https://demo.tradovateapi.com/v1/auth/accesstokenrequest`

### CME ping fails or shows N/A
- CME may block ICMP. This doesn't affect trading — it only affects the ping test.

### SSH connection refused
- Check UFW: `sudo ufw status`
- Ensure SSH is allowed: `sudo ufw allow ssh`

### Bot service won't start
- Check logs: `journalctl -u mes-bot -n 50`
- Verify the bot code exists at `/opt/mes-bot/src/main.py`
- Verify the venv is intact: `/opt/mes-bot/venv/bin/python --version`

### Out of memory (1 GB plan)
- Check usage: `free -h` and `htop`
- If needed, add swap: `sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`
- Or upgrade to the $12/mo plan (2 GB RAM)

---

## Exit Criteria

- [ ] VPS live and accessible via SSH
- [ ] Provisioning script ran successfully
- [ ] UFW firewall active (SSH only)
- [ ] Tradovate API reachable from VPS
- [ ] uvloop installed and working
- [ ] systemd service template created
