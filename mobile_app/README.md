# Mobile App (iOS) – Playing Card Detection

This folder contains a Flutter + native iOS camera scaffold for real-time playing-card detection.

## What’s Included

- Flutter UI with overlay rendering (boxes + labels + confidence)
- Native iOS `UiKitView` camera preview (AVCapture)
- 10 FPS inference throttle (preview runs full FPS)
- `EventChannel` streaming detections to Flutter
- `MethodChannel` torch toggle
- Stubbed detector class (`OnnxCardDetector`) with a mock detection to validate end-to-end UI

## Run (iOS)

From `mobile_app/card_detector`:

```bash
flutter run -d "iPhone"
```

You should see the camera preview and a single mock detection box labeled `AS`.

## Next Steps (Object Detection)

1. Export your YOLOv8 model to ONNX (e.g., 640×640 input).
2. Add ONNX Runtime to the iOS build (CocoaPods or SwiftPM).
3. Implement in `ios/Runner/AppDelegate.swift`:
   - `CVPixelBuffer` → tensor preprocessing (resize/letterbox + normalize)
   - ORT session inference
   - YOLO output decode + NMS
   - Map model coordinates → preview/view coordinates

## Channels

- `UiKitView` type: `card_detector/camera_view`
- Detections `EventChannel`: `card_detector/detections`
- Camera control `MethodChannel`: `card_detector/camera_control` (`setTorch`)

