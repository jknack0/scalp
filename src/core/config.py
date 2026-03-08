"""Bot configuration via pydantic-settings (env + YAML merge)."""

from datetime import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotConfig(BaseSettings):
    """MES scalping bot configuration.

    Load order (later overrides earlier):
    1. Field defaults (defined here)
    2. YAML config file (via from_yaml)
    3. .env file
    4. Environment variables
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Trading parameters ─────────────────────────────
    symbol: str = "MESM6"
    session_start: time = time(9, 30)
    session_end: time = time(16, 0)
    timezone: str = "America/New_York"

    # ── Risk limits ────────────────────────────────────
    max_daily_loss_usd: float = 150.0
    max_position_contracts: int = 1
    max_signals_per_day: int = 10
    slippage_assumption_ticks: float = 1.0

    # ── Tradovate credentials (from env) ───────────────
    tradovate_username: str = ""
    tradovate_password: str = ""
    tradovate_app_id: str = ""
    tradovate_app_version: str = "1.0.0"
    tradovate_cid: str = ""
    tradovate_secret: str = ""
    tradovate_demo: bool = True

    # ── Infrastructure ─────────────────────────────────
    database_url: str = ""
    databento_api_key: str = ""

    # ── Monitoring ─────────────────────────────────────
    health_port: int = 8080
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Logging ────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "logs/bot.log"

    @classmethod
    def from_yaml(
        cls, path: str | Path = "config/bot-config.yaml", **overrides: Any
    ) -> "BotConfig":
        """Load config from YAML file, then overlay env vars and overrides."""
        path = Path(path)
        yaml_data: dict[str, Any] = {}
        if path.exists():
            with open(path) as f:
                yaml_data = yaml.safe_load(f) or {}
        yaml_data.update(overrides)
        return cls(**yaml_data)
