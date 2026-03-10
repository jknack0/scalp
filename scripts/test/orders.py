"""Smoke test for Tradovate demo order placement.

Usage:
    python scripts/test/orders.py

Authenticates to Tradovate demo, finds the MES contract,
places a limit order well away from market, checks status,
then cancels it.
"""

import asyncio
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

import aiohttp

from src.core.config import BotConfig
from src.feeds.tradovate import TradovateAuth


async def main() -> None:
    config = BotConfig.from_yaml()
    if not config.tradovate_username or not config.tradovate_password:
        print("ERROR: Set TRADOVATE_USERNAME and TRADOVATE_PASSWORD in .env")
        return

    # Force demo mode
    config.tradovate_demo = True

    auth = TradovateAuth(config)
    print("Authenticating to Tradovate demo...")
    await auth.authenticate()
    print(f"  Authenticated. user_id={auth.user_id}")

    base = auth.base_url
    headers = {"Authorization": f"Bearer {auth.access_token}"}

    async with aiohttp.ClientSession(headers=headers) as session:
        # 1. Find the MES contract
        print(f"\nLooking up contract for {config.symbol}...")
        async with session.get(f"{base}/contract/find", params={"name": config.symbol}) as resp:
            if resp.status != 200:
                print(f"  Contract lookup failed (HTTP {resp.status}): {await resp.text()}")
                return
            contract = await resp.json()

        contract_id = contract["id"]
        contract_name = contract["name"]
        print(f"  Found: {contract_name} (id={contract_id})")

        # 2. Get account list
        print("\nFetching accounts...")
        async with session.get(f"{base}/account/list") as resp:
            if resp.status != 200:
                print(f"  Account list failed (HTTP {resp.status}): {await resp.text()}")
                return
            accounts = await resp.json()

        if not accounts:
            print("  No accounts found!")
            return

        account = accounts[0]
        account_id = account["id"]
        account_name = account.get("name", "unknown")
        print(f"  Using account: {account_name} (id={account_id})")

        # 3. Place a limit buy order far below market (should not fill)
        limit_price = 4000.00  # Well below any reasonable MES price
        order_payload = {
            "accountSpec": account_name,
            "accountId": account_id,
            "action": "Buy",
            "symbol": contract_name,
            "orderQty": 1,
            "orderType": "Limit",
            "price": limit_price,
            "timeInForce": "Day",
            "isAutomated": True,
        }

        print(f"\nPlacing limit BUY order: {contract_name} @ {limit_price}...")
        async with session.post(f"{base}/order/placeorder", json=order_payload) as resp:
            resp_text = await resp.text()
            if resp.status != 200:
                print(f"  Order placement failed (HTTP {resp.status}): {resp_text}")
                # Markets might be closed — that's expected
                if "market" in resp_text.lower() or "closed" in resp_text.lower():
                    print("  (Expected — futures markets are closed)")
                return
            order_result = await resp.json()

        order_id = order_result.get("orderId") or order_result.get("id")
        print(f"  Order placed! orderId={order_id}")
        print(f"  Response: {order_result}")

        # 4. Check order status
        if order_id:
            print(f"\nChecking order status...")
            async with session.get(f"{base}/order/item", params={"id": order_id}) as resp:
                if resp.status == 200:
                    order_status = await resp.json()
                    print(f"  Status: {order_status.get('ordStatus', 'unknown')}")
                else:
                    print(f"  Status check failed (HTTP {resp.status})")

            # 5. Cancel the order
            print(f"\nCancelling order {order_id}...")
            cancel_payload = {"orderId": order_id}
            async with session.post(f"{base}/order/cancelorder", json=cancel_payload) as resp:
                if resp.status == 200:
                    cancel_result = await resp.json()
                    print(f"  Cancelled! Response: {cancel_result}")
                else:
                    print(f"  Cancel failed (HTTP {resp.status}): {await resp.text()}")

        print("\nDone. Smoke test complete.")


if __name__ == "__main__":
    asyncio.run(main())
