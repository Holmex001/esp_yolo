import asyncio
import base64
import json
import threading
import time

import cv2


def build_frame_message(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return json.dumps(
        {
            "type": "frame",
            "mime": "image/jpeg",
            "data": encoded,
        },
        separators=(",", ":"),
    )


def resize_frame_for_relay(frame, max_width: int):
    if not max_width or max_width <= 0:
        return frame

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / float(width)
    target_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, target_size)


class LatestPayloadBuffer:
    def __init__(self):
        self._condition = threading.Condition()
        self._sequence = 0
        self._payload = None
        self._closed = False

    def submit(self, payload) -> int:
        with self._condition:
            self._sequence += 1
            self._payload = payload
            self._condition.notify_all()
            return self._sequence

    def wait_for_next(self, last_sequence: int, timeout: float | None = None) -> tuple[int, object]:
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while not self._closed and self._sequence <= last_sequence:
                if deadline is None:
                    remaining = None
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Timed out waiting for the next relay payload.")
                self._condition.wait(remaining)

            if self._closed:
                raise RuntimeError("Relay payload buffer has been closed.")

            return self._sequence, self._payload

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class FrameRelayClient:
    def __init__(
        self,
        url: str,
        jpeg_quality: int = 35,
        max_width: int = 480,
        min_interval: float = 0.2,
        retry_delay: float = 2.0,
        connect_timeout: float = 10.0,
    ):
        self.url = (url or "").strip()
        self.jpeg_quality = int(jpeg_quality)
        self.max_width = int(max_width)
        self.min_interval = float(min_interval)
        self.retry_delay = float(retry_delay)
        self.connect_timeout = float(connect_timeout)
        self._last_submit_at = 0.0
        self._buffer = LatestPayloadBuffer()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if not self.url or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="frame-relay")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._buffer.close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def publish_frame(self, frame) -> bool:
        if not self.url:
            return False

        now = time.monotonic()
        if self.min_interval > 0 and (now - self._last_submit_at) < self.min_interval:
            return False

        relay_frame = resize_frame_for_relay(frame, self.max_width)
        encoded, jpeg = cv2.imencode(
            ".jpg",
            relay_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not encoded:
            return False

        self._last_submit_at = now
        self._buffer.submit(jpeg.tobytes())
        return True

    def _run(self):
        asyncio.run(self._run_async())

    async def _run_async(self):
        try:
            import websockets
        except ImportError as exc:
            print(f"Relay disabled: missing websockets dependency ({exc})")
            return

        last_sequence = 0

        while not self._stop_event.is_set():
            try:
                print(f"Connecting relay to {self.url} ...")
                async with websockets.connect(
                    self.url,
                    open_timeout=self.connect_timeout,
                    max_size=8 * 1024 * 1024,
                ) as websocket:
                    print("Relay connected.")
                    while not self._stop_event.is_set():
                        try:
                            sequence, payload = await asyncio.to_thread(
                                self._buffer.wait_for_next,
                                last_sequence,
                                0.5,
                            )
                        except TimeoutError:
                            continue
                        except RuntimeError:
                            return

                        await websocket.send(payload)
                        last_sequence = sequence
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                print(f"Relay error: {exc}")
                await asyncio.sleep(self.retry_delay)
