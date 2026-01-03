import asyncio

import pytest

from precision_trigger_service import TriggerBroker


def test_broker_emits_fallback_after_two_ticks():
    async def _run() -> None:
        broker = TriggerBroker(fallback_ticks=2, tick_interval=0.01)
        received = asyncio.Queue()

        await broker.add_subscriber("test", {"BTCUSD"}, received.put)

        trigger = await broker.publish_trigger("BTCUSD", price=50000.0)
        first = await asyncio.wait_for(received.get(), timeout=1)
        assert first["type"] == "precision_trigger"
        assert first["id"] == trigger.id

        await broker.tick()
        assert received.empty()

        await broker.tick()
        fallback = await asyncio.wait_for(received.get(), timeout=1)
        assert fallback["type"] == "fallback_arm"
        assert fallback["source_event_id"] == trigger.id
        assert fallback["asset"] == "BTCUSD"

    asyncio.run(_run())


def test_broker_respects_channel_filters():
    async def _run() -> None:
        broker = TriggerBroker(fallback_ticks=1, tick_interval=0.01)
        btc_queue, eth_queue = asyncio.Queue(), asyncio.Queue()

        await broker.add_subscriber("btc", {"BTCUSD"}, btc_queue.put)
        await broker.add_subscriber("all", {"*"}, eth_queue.put)

        await broker.publish_trigger("BTCUSD", price=123.0)

        btc_msg = await asyncio.wait_for(btc_queue.get(), timeout=1)
        eth_msg = await asyncio.wait_for(eth_queue.get(), timeout=1)

        assert btc_msg["asset"] == "BTCUSD"
        assert btc_msg == eth_msg

        await broker.publish_trigger("ETHUSD", price=456.0)
        fallback = await asyncio.wait_for(eth_queue.get(), timeout=1)
        assert fallback["asset"] == "ETHUSD"

        # The BTC-only subscriber should not receive ETH events.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(btc_queue.get(), timeout=0.05)

    asyncio.run(_run())
