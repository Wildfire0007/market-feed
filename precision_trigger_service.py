"""Precision trigger websocket microservice with fallback arming.

This module exposes two layers:
- ``TriggerBroker``: in-memory broadcast/fallback engine that understands ticks.
- ``PrecisionTriggerWebsocketService``: thin websocket server around the broker.

Run the service locally:
    python precision_trigger_service.py --demo

The demo mode periodically publishes synthetic triggers so a websocket client
can observe both the real-time trigger and the automatic "arm after 2 ticks"
fallback events.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Iterable, Optional, Set

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only used for type hints
    from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


@dataclass
class TriggerEvent:
    """A precision trigger event.

    Attributes:
        id: Unique identifier for correlating fallback events.
        asset: Asset symbol (e.g. ``"BTCUSD"``) this trigger applies to.
        price: Optional price snapshot to help consumers contextualize the trigger.
        created_ts: UNIX timestamp (float seconds) when the trigger was generated.
        payload: Arbitrary metadata attached by the publisher.
    """

    id: str
    asset: str
    price: Optional[float]
    created_ts: float
    payload: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "type": "precision_trigger",
            "id": self.id,
            "asset": self.asset,
            "price": self.price,
            "created_ts": self.created_ts,
            "payload": self.payload,
        }


SubscriberFn = Callable[[Dict[str, object]], Awaitable[None]]


class TriggerBroker:
    """In-memory broadcast engine with tick-based fallback arming.

    The broker is intentionally lightweight and does not persist state; callers
    should run it alongside a durable message bus if required.
    """

    def __init__(self, *, fallback_ticks: int = 2, tick_interval: float = 1.0):
        self._subscribers: Dict[str, Dict[str, object]] = {}
        self._pending: Dict[str, Dict[str, object]] = {}
        self._fallback_ticks = fallback_ticks
        self._tick_interval = tick_interval
        self._tick_task: Optional[asyncio.Task] = None
        self._running = False

    async def add_subscriber(
        self, name: str, channels: Iterable[str], send: SubscriberFn
    ) -> None:
        """Register a subscriber interested in one or more assets.

        Channels may contain ``"*"`` to receive all events.
        """

        self._subscribers[name] = {
            "channels": set(channels),
            "send": send,
        }
        logger.info("subscriber registered", extra={"name": name, "channels": list(channels)})

    async def update_subscriptions(self, name: str, channels: Iterable[str]) -> None:
        if name not in self._subscribers:
            raise KeyError(f"unknown subscriber: {name}")
        self._subscribers[name]["channels"] = set(channels)
        logger.debug(
            "subscriber updated", extra={"name": name, "channels": list(channels)}
        )

    async def remove_subscriber(self, name: str) -> None:
        self._subscribers.pop(name, None)
        logger.info("subscriber removed", extra={"name": name})

    async def publish_trigger(
        self, asset: str, price: Optional[float] = None, payload: Optional[Dict[str, object]] = None
    ) -> TriggerEvent:
        payload = payload or {}
        trigger = TriggerEvent(
            id=str(uuid.uuid4()),
            asset=asset,
            price=price,
            created_ts=time.time(),
            payload=payload,
        )
        await self._broadcast(trigger.as_dict())
        self._pending[trigger.id] = {
            "trigger": trigger,
            "ticks": 0,
        }
        return trigger

    async def _broadcast(self, event: Dict[str, object]) -> None:
        # fan-out concurrently but wait for completion to keep ordering
        senders = [sub["send"] for sub in self._matching_subscribers(event)]
        if not senders:
            logger.debug("no subscribers for event", extra={"event": event})
            return
        await asyncio.gather(*(sender(event) for sender in senders), return_exceptions=False)

    def _matching_subscribers(self, event: Dict[str, object]):
        asset = event.get("asset")
        for sub in self._subscribers.values():
            channels: Set[str] = sub["channels"]
            if "*" in channels or asset in channels:
                yield sub

    async def tick(self) -> None:
        """Advance ticks and emit fallback "arm" events when needed."""

        expired = []
        for trigger_id, state in list(self._pending.items()):
            state["ticks"] += 1
            if state["ticks"] >= self._fallback_ticks:
                expired.append(trigger_id)

        for trigger_id in expired:
            state = self._pending.pop(trigger_id)
            trigger: TriggerEvent = state["trigger"]
            fallback_event = {
                "type": "fallback_arm",
                "source_event_id": trigger.id,
                "asset": trigger.asset,
                "created_ts": time.time(),
                "ticks_waited": state["ticks"],
                "payload": trigger.payload,
            }
            await self._broadcast(fallback_event)

    async def run_tick_loop(self) -> None:
        """Start the background tick loop until ``stop`` is called."""

        if self._running:
            return
        self._running = True
        self._tick_task = asyncio.current_task()
        try:
            while self._running:
                await asyncio.sleep(self._tick_interval)
                await self.tick()
        finally:
            self._running = False

    async def stop(self) -> None:
        self._running = False
        if self._tick_task:
            self._tick_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._tick_task
            self._tick_task = None


class PrecisionTriggerWebsocketService:
    """Thin websocket wrapper around ``TriggerBroker``."""

    def __init__(
        self,
        broker: TriggerBroker,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        fallback_ticks: int = 2,
        tick_interval: float = 1.0,
    ) -> None:
        self.broker = broker
        self.host = host
        self.port = port
        self._server = None
        self._tick_task: Optional[asyncio.Task] = None
        self._tick_interval = tick_interval
        self._fallback_ticks = fallback_ticks

    async def start(self) -> None:
        import websockets

        self._server = await websockets.serve(self._handle_client, self.host, self.port)
        self._tick_task = asyncio.create_task(self.broker.run_tick_loop())
        logger.info("precision trigger websocket service started", extra={"host": self.host, "port": self.port})

    async def _handle_client(self, websocket: "WebSocketServerProtocol") -> None:
        name = f"client-{id(websocket)}"
        await self.broker.add_subscriber(name, {"*"}, lambda event: self._send(websocket, event))
        try:
            await self._send(
                websocket,
                {
                    "type": "hello",
                    "message": "Send {\"action\":\"subscribe\",\"channels\":[...]} to filter assets.",
                    "fallback_ticks": self._fallback_ticks,
                    "tick_interval": self._tick_interval,
                },
            )
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send(websocket, {"type": "error", "message": "invalid JSON"})
                    continue

                action = msg.get("action")
                if action == "subscribe":
                    channels = msg.get("channels") or ["*"]
                    await self.broker.update_subscriptions(name, channels)
                    await self._send(
                        websocket,
                        {"type": "subscribed", "channels": channels, "fallback_ticks": self._fallback_ticks},
                    )
                elif action == "publish":
                    payload = msg.get("payload") or {}
                    await self.broker.publish_trigger(msg.get("asset", "UNKNOWN"), msg.get("price"), payload)
                else:
                    await self._send(
                        websocket,
                        {
                            "type": "error",
                            "message": "unknown action",
                            "allowed": ["subscribe", "publish"],
                        },
                    )
        finally:
            await self.broker.remove_subscriber(name)

    async def _send(self, websocket: Any, event: Dict[str, object]) -> None:
        try:
            await websocket.send(json.dumps(event))
        except Exception:  # pragma: no cover - network failures
            logger.exception("failed to send message")

    async def wait_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("service not started")
        await asyncio.Future()


async def _demo_publisher(broker: TriggerBroker, asset: str) -> None:
    """Emit a synthetic trigger every 3 seconds for demo usage."""

    while True:
        await asyncio.sleep(3)
        price = round(50000 + 1000 * (0.5 - time.time() % 1), 2)
        await broker.publish_trigger(asset=asset, price=price, payload={"source": "demo"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime precision trigger websocket service")
    parser.add_argument("--host", default="0.0.0.0", help="listen host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="listen port (default: 8765)")
    parser.add_argument(
        "--tick-interval", type=float, default=1.0, help="tick interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--fallback-ticks", type=int, default=2, help="number of ticks before fallback arm (default: 2)"
    )
    parser.add_argument("--demo", action="store_true", help="publish demo triggers")
    parser.add_argument(
        "--demo-asset",
        default="BTCUSD",
        help="asset symbol to use when demo publishing is enabled (default: BTCUSD)",
    )
    return parser


async def _run_service(args: argparse.Namespace) -> None:
    broker = TriggerBroker(fallback_ticks=args.fallback_ticks, tick_interval=args.tick_interval)
    service = PrecisionTriggerWebsocketService(
        broker,
        host=args.host,
        port=args.port,
        fallback_ticks=args.fallback_ticks,
        tick_interval=args.tick_interval,
    )
    await service.start()

    tasks = [asyncio.create_task(service.wait_forever())]
    if args.demo:
        tasks.append(asyncio.create_task(_demo_publisher(broker, args.demo_asset)))
    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()

    loop = asyncio.get_event_loop()

    for signame in {signal.SIGINT, signal.SIGTERM}:
        loop.add_signal_handler(signame, loop.stop)

    try:


        loop.run_until_complete(_run_service(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
