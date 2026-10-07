from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import typer
from rich import print
from rich.table import Table

from visionbrain.benchmark import benchmark as run_benchmark
from visionbrain.benchmark import compare_cpu_gpu
from visionbrain.calibration.chessboard import collect_and_calibrate
from visionbrain.camera.autotune import auto_tune_resolution, recommend_detector_profile
from visionbrain.camera.discovery import discover_webcams
from visionbrain.camera.opencv_source import OpenCVCamera
from visionbrain.config import CameraConfig, load_config
from visionbrain.demo import run_synthetic_e2e
from visionbrain.inference.gpu_info import detect_gpu
from visionbrain.inference.vlm import SnapshotVLM
from visionbrain.preprocess.lab import run_filter_lab
from visionbrain.quality.advisor import recommendations
from visionbrain.quality.metrics import analyze_frame_quality
from visionbrain.runtime import VisionRuntime

app = typer.Typer(add_completion=False, help="VisionBrain computer-vision engine CLI")


def parse_source(value: str):
    return int(value) if value.isdigit() else value


@app.command()
def devices(max_index: int = 6, backend: str = "auto"):
    """Discover local webcam devices."""
    found = discover_webcams(max_index=max_index, backend=backend)
    if not found:
        print("[yellow]No webcam found.[/yellow]")
        raise typer.Exit(1)
    table = Table("Index", "Backend", "Resolution", "FPS", "FOURCC")
    for device in found:
        properties = device["properties"]
        table.add_row(
            str(device["source"]),
            device["backend_name"],
            f"{int(properties['width'])}x{int(properties['height'])}",
            f"{properties['fps']:.1f}",
            device["fourcc_text"],
        )
    print(table)


@app.command()
def probe(source: str = "0", backend: str = "auto"):
    """Open an OpenCV source and print properties reported by its driver."""
    cfg = CameraConfig(source=parse_source(source), backend=backend)
    with OpenCVCamera(cfg) as camera:
        camera.warmup(10)
        print(json.dumps(camera.probe(), indent=2, ensure_ascii=False))


@app.command()
def autotune(source: str = "0", backend: str = "auto", samples: int = 20):
    """Benchmark common resolutions and select a quality/performance candidate."""
    cfg = CameraConfig(source=parse_source(source), backend=backend)
    with OpenCVCamera(cfg) as camera:
        best, results = auto_tune_resolution(camera, samples=samples)
        table = Table("Requested", "Actual", "Driver FPS", "Observed FPS", "Quality", "Score")
        for result in results:
            table.add_row(
                f"{result.requested_width}x{result.requested_height}",
                f"{result.actual_width}x{result.actual_height}",
                f"{result.actual_fps:.1f}",
                f"{result.measured_capture_fps:.1f}",
                f"{result.quality_score:.3f}",
                f"{result.score:.3f}",
            )
        print(table)
        print(f"[green]Selected:[/green] {best.actual_width}x{best.actual_height} score={best.score:.3f}")


@app.command(name="run")
def run_engine(
    config: str = "config/default.yaml",
    source: str | None = None,
    model: str | None = None,
    no_display: bool = False,
    frames: int | None = None,
):
    """Run the real-time vision loop."""
    cfg = load_config(config)
    if source is not None:
        cfg.camera.source = parse_source(source)
    if model is not None:
        cfg.detector.model = model
    if no_display:
        cfg.display.enabled = False
    result = VisionRuntime(cfg).run(max_frames=frames)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


@app.command(name="filter-lab")
def filter_lab(source: str = "0", backend: str = "auto"):
    """Compare classic vision filters live on the webcam."""
    cfg = CameraConfig(source=parse_source(source), backend=backend, width=1280, height=720, fps=30)
    run_filter_lab(cfg)


@app.command()
def diagnose(source: str = "0", backend: str = "auto", samples: int = 30):
    """Sample an OpenCV source and print capture-quality diagnostics."""
    cfg = CameraConfig(source=parse_source(source), backend=backend)
    metrics = []
    with OpenCVCamera(cfg) as camera:
        camera.warmup()
        for _ in range(samples):
            read = camera.read()
            if read.ok and read.frame is not None:
                metrics.append(analyze_frame_quality(read.frame))
        if not metrics:
            raise typer.Exit(1)
        quality = metrics[-1]
        mean_quality = sum(item.quality_score for item in metrics) / len(metrics)
        print(json.dumps({"camera": camera.probe(), "latest_quality": quality.to_dict(), "mean_quality_score": mean_quality}, indent=2, ensure_ascii=False))
        for recommendation in recommendations(quality):
            print(f"- {recommendation}")


@app.command()
def benchmark(config: str = "config/default.yaml", frames: int = 200, source: str | None = None):
    """Measure capture, preprocessing and inference latency."""
    cfg = load_config(config)
    if source is not None:
        cfg.camera.source = parse_source(source)
    result = run_benchmark(cfg, frames=frames)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


@app.command(name="gpu-info")
def gpu_info():
    """Report whether a real CUDA GPU is available via torch on this machine."""
    info = detect_gpu()
    print(json.dumps(asdict(info), indent=2, ensure_ascii=False))
    profile = recommend_detector_profile(info.cuda_available, info.device_name)
    print(json.dumps(asdict(profile), indent=2, ensure_ascii=False))


@app.command(name="benchmark-gpu")
def benchmark_gpu(
    config: str = "config/default.yaml",
    frames: int = 60,
    warmup_iters: int = 5,
    model: str | None = None,
    output: str | None = None,
):
    """Measure real CPU vs CUDA inference latency/FPS with the configured YOLO model.

    Uses the deterministic synthetic camera for both runs so the comparison isolates
    the inference device. Fails loudly if no CUDA GPU is detected instead of silently
    reporting CPU-only numbers as a GPU benchmark.
    """
    cfg = load_config(config)
    result = compare_cpu_gpu(cfg, frames=frames, warmup_iters=warmup_iters, model=model)
    payload = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    print(payload)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")


@app.command()
def calibrate(
    source: str = "0",
    backend: str = "auto",
    cols: int = 9,
    rows: int = 6,
    square_mm: float = 25.0,
    samples: int = 15,
    output: str = "outputs/calibration.yaml",
):
    """Interactive intrinsic camera calibration with a chessboard target."""
    cfg = CameraConfig(source=parse_source(source), backend=backend)
    result = collect_and_calibrate(
        cfg,
        cols=cols,
        rows=rows,
        square_size_mm=square_mm,
        required_samples=samples,
        output_path=output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def vqa(image: str, question: str, model: str = "llava-hf/llava-interleave-qwen-0.5b-hf"):
    """Ask a VLM a question about a saved frame/snapshot."""
    frame = cv2.imread(str(Path(image)))
    if frame is None:
        raise typer.BadParameter(f"Could not read image: {image}")
    print(SnapshotVLM(model=model).ask(frame, question))


@app.command()
def demo(output: str = "outputs/demo", frames: int = 24):
    """Run the offline synthetic end-to-end pipeline without webcam or model download."""
    result = run_synthetic_e2e(output, frames=frames)
    print(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def dashboard(events: str = "outputs/events/events.jsonl", port: int = 8501):
    """Start the local read-only Streamlit operations dashboard."""
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise typer.BadParameter("Install dashboard support with: pip install -e '.[dashboard]'") from exc
    app_path = Path(__file__).resolve().parents[2] / "apps" / "dashboard" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--",
        events,
    ]
    raise typer.Exit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    app()
