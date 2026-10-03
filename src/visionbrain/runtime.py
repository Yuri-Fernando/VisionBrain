from __future__ import annotations

import json
import time
from collections import deque

import cv2

from visionbrain.anomaly.motion import MotionAnomalyDetector
from visionbrain.camera.base import FrameSource
from visionbrain.camera.factory import create_frame_source
from visionbrain.config import RuntimeConfig
from visionbrain.events.dispatcher import EventDispatcher
from visionbrain.events.engine import EventEngine
from visionbrain.inference.base import DetectorBackend
from visionbrain.inference.factory import create_detector
from visionbrain.models import FrameQuality, RuntimeStats
from visionbrain.overlay import draw_overlay
from visionbrain.preprocess.pipeline import PreprocessPipeline
from visionbrain.quality.metrics import analyze_frame_quality


class VisionRuntime:
    def __init__(self, config: RuntimeConfig, *, source: FrameSource | None = None, detector: DetectorBackend | None = None, dispatcher: EventDispatcher | None = None):
        self.config = config
        self.camera = source or create_frame_source(config.camera)
        self.preprocess = PreprocessPipeline(config.preprocess)
        self.detector = detector or create_detector(config.detector)
        self.motion = MotionAnomalyDetector(config.motion_anomaly)
        self.events = EventEngine(config.events)
        self.dispatcher = dispatcher or EventDispatcher(config.events)
        self.stats = RuntimeStats()
        self.last_quality: FrameQuality | None = None
        self._fps_times = deque(maxlen=60)
        self.paused = False

    def _analyze_quality(self, frame):
        config = self.config.quality
        return analyze_frame_quality(
            frame,
            target_brightness=config.target_brightness,
            min_contrast=config.min_contrast,
            min_sharpness=config.min_sharpness,
            max_black_clip_ratio=config.max_black_clip_ratio,
            max_white_clip_ratio=config.max_white_clip_ratio,
            max_noise_score=config.max_noise_score,
        )

    def _update_fps(self) -> None:
        now = time.perf_counter()
        self._fps_times.append(now)
        if len(self._fps_times) >= 2:
            duration = self._fps_times[-1] - self._fps_times[0]
            if duration > 0:
                self.stats.fps = (len(self._fps_times) - 1) / duration

    def run(self, max_frames: int | None = None) -> dict:
        self.camera.open()
        self.camera.warmup()
        probe = self.camera.probe()
        print(json.dumps({"camera": probe}, indent=2, ensure_ascii=False))
        if self.config.display.enabled:
            print("Controls: Q/ESC quit | S snapshot | SPACE pause")
        try:
            while max_frames is None or self.stats.frame_index < max_frames:
                if self.paused:
                    key = cv2.waitKey(30) & 0xFF if self.config.display.enabled else ord(" ")
                    if key in (ord(" "), ord("p")):
                        self.paused = False
                    elif key in (ord("q"), 27):
                        break
                    continue

                read = self.camera.read()
                if not read.ok or read.frame is None:
                    continue
                frame = read.frame
                self.stats.frame_index += 1

                quality_config = self.config.quality
                if quality_config.enabled and (
                    self.last_quality is None
                    or self.stats.frame_index % max(quality_config.every_n_frames, 1) == 0
                ):
                    started = time.perf_counter()
                    self.last_quality = self._analyze_quality(frame)
                    self.stats.quality_ms = (time.perf_counter() - started) * 1000

                started = time.perf_counter()
                processed = self.preprocess.apply(frame, self.last_quality)
                self.stats.preprocessing_ms = (time.perf_counter() - started) * 1000
                inference = self.detector.infer(processed)
                self.stats.inference_ms = inference.latency_ms
                self.stats.detector_count = len(inference.detections)
                self.stats.active_tracks = len({item.track_id for item in inference.detections if item.track_id is not None})

                motion = self.motion.update(processed)
                emitted = self.events.evaluate(
                    frame_index=self.stats.frame_index,
                    frame_shape=processed.shape,
                    detections=inference.detections,
                    quality=self.last_quality,
                    motion_anomalous=motion.anomalous,
                    motion_ratio=motion.motion_ratio,
                )
                self.stats.events_emitted += len(emitted)
                self._update_fps()
                visual = draw_overlay(
                    processed,
                    inference.detections,
                    self.stats,
                    self.config.display,
                    quality=self.last_quality,
                    zones=self.config.events.zones,
                    motion_ratio=motion.motion_ratio,
                    events=emitted,
                )
                for event in emitted:
                    self.dispatcher.emit(event, visual)
                    print(json.dumps(event.to_dict(), ensure_ascii=False))

                if self.config.display.enabled:
                    show = visual
                    max_width = self.config.display.resize_max_width
                    if max_width and show.shape[1] > max_width:
                        scale = max_width / show.shape[1]
                        show = cv2.resize(show, None, fx=scale, fy=scale)
                    cv2.imshow(self.config.display.window_name, show)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key in (ord(" "), ord("p")):
                        self.paused = True
                    if key == ord("s"):
                        path = f"outputs/snapshots/manual_{self.stats.frame_index}.jpg"
                        cv2.imwrite(path, visual)
                        print(f"Saved {path}")
        finally:
            self.camera.release()
            self.dispatcher.close()
            if self.config.display.enabled:
                cv2.destroyAllWindows()

        return {
            "frames": self.stats.frame_index,
            "fps": self.stats.fps,
            "inference_ms": self.stats.inference_ms,
            "events_emitted": self.stats.events_emitted,
            "camera": probe,
        }
