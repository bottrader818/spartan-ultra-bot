import websocket
import threading
import time
import json
import logging
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto
import queue
from collections import deque

class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()

@dataclass
class StreamMessage:
    data: Any
    timestamp: float
    sequence: int
    channel: str
    raw: Optional[str] = None

class DataStreamer:
    def __init__(self, url: str, max_reconnect_attempts: int = 5, reconnect_interval: float = 3.0,
                 heartbeat_interval: float = 30.0, message_buffer_size: int = 1000):
        self.url = url
        self.ws: Optional[websocket.WebSocketApp] = None
        self.thread: Optional[threading.Thread] = None
        self.state = ConnectionState.DISCONNECTED

        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_attempts = 0
        self.last_message_time = 0

        self.message_queue = queue.Queue(maxsize=message_buffer_size)
        self.message_buffer = deque(maxlen=message_buffer_size)
        self.sequence_counter = 0
        self.subscriptions = set()

        self._message_callbacks = []
        self._error_callbacks = []
        self._state_change_callbacks = []

        self.metrics = {
            'messages_received': 0,
            'messages_processed': 0,
            'connection_duration': 0.0,
            'last_error': None,
            'message_rate': 0.0,
            'reconnects': 0
        }

        self._heartbeat_timer = None
        self._should_run = threading.Event()

    def _update_state(self, new_state: ConnectionState) -> None:
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            for cb in self._state_change_callbacks:
                try:
                    cb(old_state, new_state)
                except Exception as e:
                    self.logger.error(f"State change callback failed: {e}", exc_info=True)

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        self._update_state(ConnectionState.CONNECTED)
        self.reconnect_attempts = 0
        self.last_message_time = time.time()
        self._start_heartbeat()
        self._resubscribe()
        self.metrics['reconnects'] += 1
        self.connection_start_time = time.time()

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        try:
            self.last_message_time = time.time()
            self.metrics['messages_received'] += 1

            try:
                data = json.loads(message)
                channel = data.get('channel', 'unknown')
            except json.JSONDecodeError:
                data = message
                channel = 'raw'

            msg = StreamMessage(
                data=data,
                timestamp=time.time(),
                sequence=self.sequence_counter,
                channel=channel,
                raw=message
            )
            self.sequence_counter += 1
            self.message_buffer.append(msg)
            try:
                self.message_queue.put_nowait(msg)
            except queue.Full:
                self.message_queue.get_nowait()
                self.message_queue.put_nowait(msg)

            for cb in self._message_callbacks:
                try:
                    cb(msg)
                except Exception as e:
                    self.logger.error(f"Callback failed: {e}", exc_info=True)

            self.metrics['messages_processed'] += 1
            self._update_message_rate()
        except Exception as e:
            self.logger.error(f"Message handling failed: {e}", exc_info=True)
            self._notify_error(e)

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        self.logger.error(f"WebSocket error: {error}", exc_info=True)
        self._update_state(ConnectionState.ERROR)
        self.metrics['last_error'] = str(error)
        self._notify_error(error)

    def _on_close(self, ws: websocket.WebSocketApp, code, msg) -> None:
        self.logger.warning(f"WebSocket closed: {code} - {msg}")
        self._update_state(ConnectionState.DISCONNECTED)
        self._stop_heartbeat()
        if hasattr(self, 'connection_start_time'):
            self.metrics['connection_duration'] += time.time() - self.connection_start_time
        if self._should_run.is_set():
            self._schedule_reconnect()

    def _start_heartbeat(self) -> None:
        self._stop_heartbeat()
        self._heartbeat_timer = threading.Timer(self.heartbeat_interval, self._check_heartbeat)
        self._heartbeat_timer.daemon = True
        self._heartbeat_timer.start()

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()

    def _check_heartbeat(self) -> None:
        if (time.time() - self.last_message_time) > self.heartbeat_interval * 2:
            self.logger.warning("No heartbeat received. Reconnecting...")
            self._reconnect()
        else:
            self._start_heartbeat()

    def _schedule_reconnect(self) -> None:
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            self._update_state(ConnectionState.RECONNECTING)
            threading.Timer(self.reconnect_interval, self._reconnect).start()
        else:
            self.logger.error("Max reconnect attempts reached.")
            self._update_state(ConnectionState.ERROR)

    def _reconnect(self) -> None:
        if self.state == ConnectionState.CONNECTED:
            return
        self.logger.info("Reconnecting WebSocket...")
        self._update_state(ConnectionState.CONNECTING)
        self._disconnect()
        self._connect()

    def _connect(self) -> None:
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.thread.start()
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}", exc_info=True)
            self._update_state(ConnectionState.ERROR)
            self._schedule_reconnect()

    def _disconnect(self) -> None:
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                self.logger.error(f"Disconnect error: {e}", exc_info=True)

    def _resubscribe(self) -> None:
        for channel in self.subscriptions:
            self.subscribe(channel)

    def subscribe(self, channel: str, params: Optional[Dict] = None) -> None:
        msg = {
            "action": "subscribe",
            "channel": channel
        }
        if params:
            msg["params"] = params
        self.subscriptions.add(channel)
        if self.state == ConnectionState.CONNECTED:
            try:
                self.ws.send(json.dumps(msg))
                self.logger.debug(f"Subscribed to {channel}")
            except Exception as e:
                self.logger.error(f"Subscription failed: {e}", exc_info=True)

    def unsubscribe(self, channel: str) -> None:
        msg = {
            "action": "unsubscribe",
            "channel": channel
        }
        self.subscriptions.discard(channel)
        if self.state == ConnectionState.CONNECTED:
            try:
                self.ws.send(json.dumps(msg))
            except Exception as e:
                self.logger.error(f"Unsubscribe failed: {e}", exc_info=True)

    def start(self) -> None:
        if not self._should_run.is_set():
            self._should_run.set()
            self._connect()

    def stop(self) -> None:
        self._should_run.clear()
        self._disconnect()
        self._stop_heartbeat()
        self._update_state(ConnectionState.DISCONNECTED)

    def get_message(self, timeout: Optional[float] = None) -> Optional[StreamMessage]:
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def add_message_callback(self, cb: Callable[[StreamMessage], None]) -> None:
        self._message_callbacks.append(cb)

    def add_error_callback(self, cb: Callable[[Exception], None]) -> None:
        self._error_callbacks.append(cb)

    def add_state_change_callback(self, cb: Callable[[ConnectionState, ConnectionState], None]) -> None:
        self._state_change_callbacks.append(cb)

    def _notify_error(self, error: Exception) -> None:
        for cb in self._error_callbacks:
            try:
                cb(error)
            except Exception as e:
                self.logger.error(f"Error callback failed: {e}", exc_info=True)

    def _update_message_rate(self) -> None:
        now = time.time()
        if len(self.message_buffer) > 1:
            time_span = now - self.message_buffer[0].timestamp
            if time_span > 0:
                self.metrics['message_rate'] = len(self.message_buffer) / time_span

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()

    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
