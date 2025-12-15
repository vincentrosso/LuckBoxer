# card_detector

Flutter iOS app scaffold for real-time playing card detection.

## Getting Started

- Camera preview is a native iOS `UiKitView` (`AVCaptureSession`).
- Detections stream over an `EventChannel` and render as Flutter overlays.
- ONNX inference is stubbed (mock detection) and ready to be wired.

Run:

```bash
flutter run -d "iPhone"
```

Project notes: `mobile_app/README.md`
