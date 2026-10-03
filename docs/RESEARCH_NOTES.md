# Research Notes — October 2026

## Tracking

Current Ultralytics documentation exposes multiple trackers including ByteTrack and BoT-SORT, and newer alternatives such as OC-SORT, Deep OC-SORT, FastTracker and TrackTrack. ByteTrack is a sensible low-overhead baseline for a fixed webcam. BoT-SORT adds camera-motion compensation and optional ReID, making it more appropriate when camera movement or identity switches are significant.

Source: https://docs.ultralytics.com/modes/track/

## Camera properties

OpenCV exposes generic properties including frame width/height/FPS, exposure, gain, autofocus, auto exposure, auto white balance and white-balance temperature. Support is camera/backend-dependent; therefore the engine probes the observed value after configuration rather than assuming success.

Source: https://docs.opencv.org/4.12.0/d4/d15/group__videoio__flags__base.html

## Preprocessing

Gaussian, median and bilateral filters remain standard image-smoothing primitives. VisionBrain keeps them recipe-driven because the best filter depends on the defect/task: aggressive smoothing can suppress small defect cues.

Source: https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html

## Geometric calibration

OpenCV's calibration workflow estimates camera intrinsics/distortion from multiple calibration-pattern observations and evaluates reprojection error. Diverse board poses and working distances are important for stable calibration.

Sources:
- https://docs.opencv.org/4.12.0/dc/dbb/tutorial_py_calibration.html
- https://docs.opencv.org/doc/doxygen/html/d4/d93/group__calib.html

## Industrial acquisition

Harvester provides Python access to GenICam devices through GenTL Producers. This is an appropriate vendor-neutral interface boundary, but the deployment still depends on a manufacturer's CTI producer and pixel/trigger/network configuration.

Source: https://harvesters.readthedocs.io/en/stable/TUTORIAL.html

## Learned anomaly detection

Anomalib provides a production-oriented toolkit for training and deploying visual anomaly-detection models. It belongs in an offline train/evaluate/export pipeline, while the current MOG2 component is only a no-training scene/motion baseline.

Source: https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html

## VQA / VLM

Visual question answering combines image and natural-language question input. Modern image-text-to-text pipelines generalize this into VLM inference. VisionBrain treats VLM calls as event-triggered snapshot analysis, not as the primary real-time detector.

Sources:
- https://huggingface.co/docs/transformers/main/tasks/visual_question_answering
- https://huggingface.co/docs/transformers/main/tasks/image_text_to_text

## Licensing

OpenCV 4.5+ is Apache 2.0. Ultralytics currently offers an AGPL-3.0 open-source route and an Enterprise commercial route. Commercial SaaS architecture therefore needs an explicit licensing decision before embedding Ultralytics in a proprietary product.

Sources:
- https://opencv.org/license/
- https://docs.ultralytics.com/
