import asyncio
import base64
import json
import threading

import cv2
import numpy as np


def decode_frame_message(message) -> bytes:
    if isinstance(message, (bytes, bytearray, memoryview)):
        return bytes(message)

    if isinstance(message, str):
        parsed = json.loads(message)
        if parsed.get("type") != "frame" or parsed.get("mime") != "image/jpeg" or not parsed.get("data"):
            raise ValueError("Unsupported raw-frame payload.")
        return base64.b64decode(parsed["data"], validate=True)

    raise ValueError("Unsupported raw-frame payload type.")


class LatestCloudFrameReader:
    def __init__(self, url: str, timeout: float, retry_delay: float = 1.0):
        self.url = url
        self.timeout = float(timeout)
        self.retry_delay = float(retry_delay)
        self._event = threading.Event()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_index = 0
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="cloud-raw-reader")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._event.set()
        self._thread.join(timeout=2)

    def wait_for_frame(self, last_index: int, timeout: float):
        while not self._stop_event.is_set():
            with self._lock:
                if self._latest_index > last_index:
                    return self._latest_index, self._latest_frame.copy()
                if self._error is not None:
                    raise RuntimeError(f"Cloud raw reader stopped: {self._error}")
            if not self._event.wait(timeout):
                raise TimeoutError("Timed out waiting for the next cloud raw frame.")
            self._event.clear()
        raise RuntimeError("Cloud raw reader stopped.")

    def _publish(self, frame):
        with self._lock:
            self._latest_index += 1
            self._latest_frame = frame
        self._event.set()

    def _run(self):
        asyncio.run(self._run_async())

    async def _run_async(self):
        try:
            import websockets
        except ImportError as exc:
            with self._lock:
                self._error = exc
            self._event.set()
            return

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(
                    self.url,
                    open_timeout=self.timeout,
                    max_size=8 * 1024 * 1024,
                ) as websocket:
                    async for message in websocket:
                        if self._stop_event.is_set():
                            break
                        try:
                            jpeg_bytes = decode_frame_message(message)
                            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                        except Exception as exc:
                            print(f"Skipping malformed cloud raw frame: {exc}")
                            continue
                        if frame is not None:
                            self._publish(frame)
                        else:
                            print("Skipping undecodable cloud raw frame.")
                if not self._stop_event.is_set():
                    print("Cloud raw reader disconnected, reconnecting...")
                    await asyncio.sleep(self.retry_delay)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                print(f"Cloud raw reader error: {exc}")
                await asyncio.sleep(self.retry_delay)
