import argparse
import json
import pathlib
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit

import cv2
import face_recognition
import numpy as np
import torch
from ultralytics import YOLO

from raw_frame_client import LatestCloudFrameReader
from video_relay import FrameRelayClient


DEFAULT_STREAM_URL = "http://192.168.3.200/stream"
DEFAULT_RAW_VIEW_URL = "ws://8.156.86.59/ws/raw-view"
DEFAULT_RELAY_URL = "ws://8.156.86.59/ws/push?token=2301dd693ce4d6683c167f66dc1e8b7b5b0d8c207bbd2b3a"
WINDOW_NAME = "YOLO + Face Recognition"
BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH_CPU = str(BASE_DIR / "yolov8n.pt")
MODEL_PATH_GPU = str(BASE_DIR / "yolov8m.pt")
KNOWN_FACE_PATH = str(BASE_DIR / "known_face.jpg")
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
LABEL_CONFIG_PATH = BASE_DIR / "label_config.json"
DISPLAY_MAX_WIDTH = 1280
INFERENCE_MAX_WIDTH = 960
FACE_RECOGNITION_SCALE = 1
FACE_TOLERANCE = 0.5
FACE_MEAN_TOLERANCE = 0.50
FACE_TOP_K = 3
FACE_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
REFERENCE_NUM_JITTERS = 5
PERSON_BOX_PADDING_RATIO = 0.15
MIN_PERSON_BOX_SIZE = 50
DEFAULT_YOLO_LABEL_CLASSES = ("person",)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO and face recognition on the ESP32 camera stream."
    )
    parser.add_argument(
        "--source-mode",
        choices=("lan", "cloud-raw"),
        default="cloud-raw",
        help="Read from the ESP32 over LAN or from the cloud raw-frame relay.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_STREAM_URL,
        help=f"ESP32 stream URL, default: {DEFAULT_STREAM_URL}",
    )
    parser.add_argument(
        "--raw-view-url",
        default=DEFAULT_RAW_VIEW_URL,
        help=f"Cloud raw-frame websocket URL, default: {DEFAULT_RAW_VIEW_URL}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP socket timeout in seconds.",
    )
    parser.add_argument(
        "--mode",
        choices=("stream", "snapshot", "auto"),
        default="auto",
        help="Use persistent MJPEG stream, /capture polling, or auto fallback.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Delay between /capture requests in snapshot mode.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Process one frame every N+1 frames to improve responsiveness.",
    )
    parser.add_argument(
        "--face-every",
        type=int,
        default=2,
        help="Run face recognition every N processed frames.",
    )
    parser.add_argument(
        "--relay-url",
        default=DEFAULT_RELAY_URL,
        help=f"Optional WSS relay push endpoint for annotated frames, default: {DEFAULT_RELAY_URL}",
    )
    parser.add_argument(
        "--relay-jpeg-quality",
        type=int,
        default=45,
        help="JPEG quality used when pushing annotated frames to the relay.",
    )
    parser.add_argument(
        "--relay-max-width",
        type=int,
        default=480,
        help="Maximum width of annotated frames pushed to the relay.",
    )
    parser.add_argument(
        "--relay-min-interval",
        type=float,
        default=0.1,
        help="Minimum interval in seconds between relay preview frames.",
    )
    parser.add_argument(
        "--yolo-label-classes",
        default=None,
        help="Comma-separated YOLO classes to draw, use 'none' to disable YOLO labels.",
    )
    return parser.parse_args()


def open_url(url, timeout):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Python-ESP32-YOLO-Viewer/1.0",
            "Cache-Control": "no-cache",
        },
    )
    return urllib.request.urlopen(request, timeout=timeout)


def build_stream_urls(url):
    split = urlsplit(url)
    if not split.scheme or not split.netloc:
        return [url]

    host = split.hostname or ""
    path = split.path or "/stream"
    candidates = [url]

    if host and split.port == 81 and path == "/stream":
        candidates.append(urlunsplit((split.scheme, host, "/stream", "", "")))

    return candidates


def build_capture_url(url):
    split = urlsplit(url)
    if not split.scheme or not split.netloc:
        return url

    netloc = split.netloc
    if split.port == 81 and split.hostname:
        netloc = split.hostname

    return urlunsplit((split.scheme, netloc, "/capture", "", ""))


def iter_mjpeg_frames(stream):
    buffer = bytearray()

    while True:
        chunk = stream.read(4096)
        if not chunk:
            raise ConnectionError("Stream closed by remote host.")

        buffer.extend(chunk)

        while True:
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start + 2)
            if start == -1 or end == -1:
                if len(buffer) > 1_000_000:
                    del buffer[:-4096]
                break

            jpg = bytes(buffer[start : end + 2])
            del buffer[: end + 2]

            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


def snapshot_frames(url, timeout, interval):
    capture_url = build_capture_url(url)
    print(f"Using snapshot polling: {capture_url}")

    while True:
        with open_url(capture_url, timeout) as response:
            jpg = response.read()

        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode JPEG frame from /capture.")

        yield frame
        time.sleep(interval)


def stream_frames(url, timeout):
    for candidate_url in build_stream_urls(url):
        try:
            print(f"Connecting to {candidate_url} ...")
            with open_url(candidate_url, timeout) as response:
                print("Stream connected.")
                yield from iter_mjpeg_frames(response)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout) as exc:
            print(f"Stream error from {candidate_url}: {exc}")

    raise ConnectionError("Unable to open any ESP32 stream URL.")


def frame_source(url, timeout, mode, interval):
    if mode == "stream":
        while True:
            yield from stream_frames(url, timeout)
            print("Reconnecting stream in 2 seconds...")
            time.sleep(2)
    elif mode == "snapshot":
        while True:
            try:
                yield from snapshot_frames(url, timeout, interval)
            except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout, ValueError) as exc:
                print(f"Snapshot error: {exc}")
                print("Retrying in 1 second...")
                time.sleep(1)
    else:
        while True:
            try:
                yield from stream_frames(url, timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout):
                print("Streaming is unstable, switching to snapshot polling.")
                try:
                    yield from snapshot_frames(url, timeout, interval)
                except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout, ValueError) as exc:
                    print(f"Snapshot error: {exc}")
                    print("Retrying in 1 second...")
                    time.sleep(1)


class LatestFrameReader:
    def __init__(self, url, timeout, mode, interval):
        self.url = url
        self.timeout = timeout
        self.mode = mode
        self.interval = interval
        self._latest_frame = None
        self._latest_index = 0
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop_event = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._event.set()
        self._thread.join(timeout=2)

    def wait_for_frame(self, last_index, timeout=5.0):
        while not self._stop_event.is_set():
            with self._lock:
                if self._latest_index > last_index:
                    return self._latest_index, self._latest_frame.copy()
                error = self._error

            if error is not None:
                raise RuntimeError(f"Frame reader stopped: {error}")

            if not self._event.wait(timeout):
                raise TimeoutError("Timed out waiting for the next ESP32 frame.")
            self._event.clear()

        raise RuntimeError("Frame reader was stopped.")

    def _publish_frame(self, frame):
        with self._lock:
            self._latest_index += 1
            self._latest_frame = frame
        self._event.set()

    def _run(self):
        try:
            for frame in frame_source(self.url, self.timeout, self.mode, self.interval):
                if self._stop_event.is_set():
                    break
                self._publish_frame(frame)
        except Exception as exc:
            with self._lock:
                self._error = exc
            self._event.set()


def infer_face_name(image_path):
    if image_path == pathlib.Path(KNOWN_FACE_PATH):
        return "admin"

    if KNOWN_FACES_DIR in image_path.parents and image_path.parent != KNOWN_FACES_DIR:
        return image_path.parent.name

    stem = image_path.stem
    for separator in ("_", "-"):
        if separator in stem:
            return stem.split(separator, 1)[0]
    return stem


def iter_known_face_images():
    image_paths = []
    seen = set()

    if pathlib.Path(KNOWN_FACE_PATH).exists():
        legacy_path = pathlib.Path(KNOWN_FACE_PATH)
        image_paths.append(legacy_path)
        seen.add(legacy_path.resolve())

    if KNOWN_FACES_DIR.exists():
        for path in sorted(KNOWN_FACES_DIR.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in FACE_IMAGE_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            image_paths.append(path)
            seen.add(resolved)

    return image_paths


def normalize_face_image(image):
    if image.ndim != 3 or image.shape[2] != 3:
        return image

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    y_channel = cv2.equalizeHist(y_channel)
    normalized = cv2.merge((y_channel, cr_channel, cb_channel))
    return cv2.cvtColor(normalized, cv2.COLOR_YCrCb2RGB)


def build_reference_variants(image):
    variants = [image]
    variants.append(np.ascontiguousarray(image[:, ::-1]))
    variants.append(normalize_face_image(image))
    variants.append(np.ascontiguousarray(normalize_face_image(image)[:, ::-1]))
    return variants


def extract_best_face_encoding(image, num_jitters):
    face_locations = face_recognition.face_locations(
        image,
        number_of_times_to_upsample=1,
    )
    if not face_locations:
        return None

    best_location = max(
        face_locations,
        key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
    )
    encodings = face_recognition.face_encodings(
        image,
        [best_location],
        num_jitters=num_jitters,
    )
    if not encodings:
        return None
    return encodings[0]


def load_face_database():
    print("Loading known face database...")
    known_encodings = []
    known_names = []
    image_paths = iter_known_face_images()

    if not image_paths:
        raise RuntimeError(
            "No known face images found. Add known_face.jpg or images under known_faces/."
        )

    for image_path in image_paths:
        name = infer_face_name(image_path)
        known_image = face_recognition.load_image_file(str(image_path))
        sample_count = 0

        for variant in build_reference_variants(known_image):
            encoding = extract_best_face_encoding(variant, REFERENCE_NUM_JITTERS)
            if encoding is None:
                continue

            known_encodings.append(encoding)
            known_names.append(name)
            sample_count += 1

        if sample_count == 0:
            print(f"Skipping {image_path.name}: failed to extract any face encoding.")
            continue

        print(f"Loaded face sample: {image_path.name} -> {name} ({sample_count} encodings)")

    if not known_encodings:
        raise RuntimeError(
            "Known face database is empty after loading. "
            "Use clear photos with one visible face."
        )

    print(f"Known face database loaded: {len(known_encodings)} samples.")
    return known_encodings, known_names


def resize_by_width(frame, max_width):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0

    scale = max_width / float(width)
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
    return resized, scale


def overlay_style(frame):
    height, width = frame.shape[:2]
    short_side = min(height, width)
    font_scale = max(0.35, short_side / 900.0)
    thickness = 1 if short_side < 720 else 2
    line_width = 1 if short_side < 720 else 2
    label_height = max(18, int(24 * font_scale))
    return font_scale, thickness, line_width, label_height


def parse_label_class_filter(raw_value):
    if raw_value is None:
        return set(DEFAULT_YOLO_LABEL_CLASSES)

    value = raw_value.strip().lower()
    if not value:
        return set(DEFAULT_YOLO_LABEL_CLASSES)
    if value == "none":
        return set()

    return {
        item.strip().lower()
        for item in raw_value.split(",")
        if item.strip()
    }


def load_label_config():
    if not LABEL_CONFIG_PATH.exists():
        return {}

    with LABEL_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise RuntimeError(f"Invalid config format in {LABEL_CONFIG_PATH}. Expected a JSON object.")

    return config


def resolve_yolo_label_classes(args):
    if args.yolo_label_classes is not None:
        return parse_label_class_filter(args.yolo_label_classes)

    config = load_label_config()
    label_classes = config.get("yolo_label_classes", list(DEFAULT_YOLO_LABEL_CLASSES))

    if label_classes is None:
        return set()
    if not isinstance(label_classes, list):
        raise RuntimeError(
            f"Invalid yolo_label_classes in {LABEL_CONFIG_PATH}. Expected a JSON array."
        )

    return {
        str(item).strip().lower()
        for item in label_classes
        if str(item).strip()
    }


def resolve_yolo_detect_class_ids(model, allowed_classes):
    if not allowed_classes:
        return []

    class_ids = []
    for class_id, class_name in model.names.items():
        if str(class_name).lower() in allowed_classes:
            class_ids.append(int(class_id))
    return sorted(class_ids)


def draw_yolo_detections(frame, result, allowed_classes):
    if not allowed_classes:
        return

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or boxes.cls is None:
        return

    confidences = boxes.conf.tolist() if boxes.conf is not None else [None] * len(boxes.cls)
    font_scale, thickness, line_width, label_height = overlay_style(frame)

    for xyxy, class_id, confidence in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), confidences):
        class_name = result.names.get(int(class_id), str(int(class_id))).lower()
        if class_name not in allowed_classes:
            continue

        x1, y1, x2, y2 = [int(v) for v in xyxy]
        color = (255, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)
        label = class_name if confidence is None else f"{class_name} {float(confidence):.2f}"
        label_top = max(0, y1 - label_height)
        cv2.rectangle(frame, (x1, label_top), (x2, y1), color, cv2.FILLED)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )


def expand_box(box, frame_shape, padding_ratio):
    x1, y1, x2, y2 = box
    height, width = frame_shape[:2]
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def iter_person_boxes(result, frame_shape):
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or boxes.cls is None:
        return

    class_ids = boxes.cls.tolist()
    xyxy_boxes = boxes.xyxy.tolist()

    for class_id, xyxy in zip(class_ids, xyxy_boxes):
        class_name = result.names.get(int(class_id), str(int(class_id)))
        if class_name != "person":
            continue

        x1, y1, x2, y2 = [int(v) for v in xyxy]
        if x2 - x1 < MIN_PERSON_BOX_SIZE or y2 - y1 < MIN_PERSON_BOX_SIZE:
            continue

        yield expand_box((x1, y1, x2, y2), frame_shape, PERSON_BOX_PADDING_RATIO)


def recognize_faces(frame, known_face_encodings, known_face_names, origin=(0, 0)):
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=FACE_RECOGNITION_SCALE,
        fy=FACE_RECOGNITION_SCALE,
    )
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_small_frame = normalize_face_image(rgb_small_frame)

    face_locations = face_recognition.face_locations(
        rgb_small_frame,
        number_of_times_to_upsample=2,
    )
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations,
        num_jitters=1,
    )
    face_results = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        face_distance = None
        face_mean_distance = None

        if known_face_encodings:
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            grouped_distances = {}
            for known_name, distance in zip(known_face_names, distances):
                grouped_distances.setdefault(known_name, []).append(float(distance))

            best_name = None
            best_min_distance = None
            best_mean_distance = None

            for known_name, name_distances in grouped_distances.items():
                sorted_distances = sorted(name_distances)
                min_distance = sorted_distances[0]
                mean_distance = float(np.mean(sorted_distances[:FACE_TOP_K]))

                if best_min_distance is None or min_distance < best_min_distance:
                    best_name = known_name
                    best_min_distance = min_distance
                    best_mean_distance = mean_distance

            face_distance = best_min_distance
            face_mean_distance = best_mean_distance
            if (
                best_name is not None
                and best_min_distance is not None
                and best_mean_distance is not None
                and best_min_distance <= FACE_TOLERANCE
                and best_mean_distance <= FACE_MEAN_TOLERANCE
            ):
                name = best_name

        face_results.append(
            {
                "box": (
                    int(top / FACE_RECOGNITION_SCALE) + origin[1],
                    int(right / FACE_RECOGNITION_SCALE) + origin[0],
                    int(bottom / FACE_RECOGNITION_SCALE) + origin[1],
                    int(left / FACE_RECOGNITION_SCALE) + origin[0],
                ),
                "name": name,
                "distance": face_distance,
                "mean_distance": face_mean_distance,
            }
        )

    return face_results


def recognize_faces_in_person_boxes(frame, result, known_face_encodings, known_face_names):
    face_results = []

    for x1, y1, x2, y2 in iter_person_boxes(result, frame.shape):
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue
        crop_results = recognize_faces(
            person_crop,
            known_face_encodings,
            known_face_names,
            origin=(x1, y1),
        )
        face_results.extend(crop_results)

    return face_results


def draw_faces(annotated_frame, face_results):
    font_scale, thickness, _, label_height = overlay_style(annotated_frame)

    for result in face_results:
        top, right, bottom, left = result["box"]
        name = result["name"]
        distance = result["distance"]
        mean_distance = result["mean_distance"]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, thickness)
        label_top = max(0, bottom - label_height)
        cv2.rectangle(annotated_frame, (left, label_top), (right, bottom), color, cv2.FILLED)
        if distance is None:
            label = name
        elif mean_distance is None:
            label = f"{name} {distance:.2f}"
        else:
            label = f"{name} {distance:.2f}/{mean_distance:.2f}"
        cv2.putText(
            annotated_frame,
            label,
            (left + 4, bottom - 5),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )


def detect_device():
    """Detect if CUDA GPU is available and return device name and model path."""
    if torch.cuda.is_available():
        device = "cuda"
        model_path = MODEL_PATH_GPU
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
        print(f"Using GPU with model: {model_path}")
    else:
        device = "cpu"
        model_path = MODEL_PATH_CPU
        print("No GPU detected, using CPU")
        print(f"Using CPU with model: {model_path}")
    return device, model_path


def main():
    args = parse_args()
    socket.setdefaulttimeout(args.timeout)
    yolo_label_classes = resolve_yolo_label_classes(args)

    device, model_path = detect_device()
    print("Loading YOLO model...")
    model = YOLO(model_path)
    yolo_detect_class_ids = resolve_yolo_detect_class_ids(model, yolo_label_classes)
    print(
        "YOLO detect classes: "
        + (", ".join(model.names[class_id] for class_id in yolo_detect_class_ids) if yolo_detect_class_ids else "none")
    )
    known_face_encodings, known_face_names = load_face_database()
    if args.source_mode == "cloud-raw":
        reader = LatestCloudFrameReader(args.raw_view_url, args.timeout)
    else:
        reader = LatestFrameReader(args.url, args.timeout, args.mode, args.interval)
    relay_client = None
    if args.relay_url:
        relay_client = FrameRelayClient(
            args.relay_url,
            jpeg_quality=args.relay_jpeg_quality,
            max_width=args.relay_max_width,
            min_interval=args.relay_min_interval,
            connect_timeout=args.timeout,
        )
        relay_client.start()
        print(f"Relay push enabled: {args.relay_url}")
    reader.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    prev_time = time.time()
    frame_index = 0
    latest_reader_index = 0
    last_face_results = []

    try:
        while True:
            try:
                latest_reader_index, frame = reader.wait_for_frame(latest_reader_index, timeout=args.timeout)
            except TimeoutError as exc:
                print(exc)
                continue
            frame_index += 1
            if args.skip_frames and frame_index % (args.skip_frames + 1) != 1:
                continue

            display_frame, _ = resize_by_width(frame, DISPLAY_MAX_WIDTH)
            inference_frame, _ = resize_by_width(display_frame, INFERENCE_MAX_WIDTH)

            results = model.predict(
                source=inference_frame,
                device=device,
                classes=yolo_detect_class_ids,
                conf=args.conf,
                verbose=False,
            )
            annotated_frame = inference_frame.copy()
            draw_yolo_detections(annotated_frame, results[0], yolo_label_classes)

            if args.face_every <= 1 or frame_index % args.face_every == 1:
                last_face_results = recognize_faces_in_person_boxes(
                    inference_frame,
                    results[0],
                    known_face_encodings,
                    known_face_names,
                )
            draw_faces(annotated_frame, last_face_results)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Reader frame: {latest_reader_index}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            if relay_client is not None:
                relay_client.publish_frame(annotated_frame)

            cv2.imshow(WINDOW_NAME, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if relay_client is not None:
            relay_client.stop()
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
