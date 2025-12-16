import AVFoundation
import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    if let registrar = self.registrar(forPlugin: "CardCameraPlatformView") {
      let factory = CardCameraPlatformViewFactory()
      registrar.register(factory, withId: "card_detector/camera_view")

      let messenger = registrar.messenger()

      configureOnnxDetector(registrar: registrar)

      let eventChannel = FlutterEventChannel(
        name: "card_detector/detections",
        binaryMessenger: messenger
      )
      eventChannel.setStreamHandler(DetectionEventStreamHandler.shared)

      let methodChannel = FlutterMethodChannel(
        name: "card_detector/camera_control",
        binaryMessenger: messenger
      )
      methodChannel.setMethodCallHandler { call, result in
        switch call.method {
        case "setTorch":
          guard let args = call.arguments as? [String: Any],
                let enabled = args["enabled"] as? Bool
          else {
            result(
              FlutterError(
                code: "invalid_args",
                message: "Expected { enabled: bool }",
                details: nil
              )
            )
            return
          }
          CardCameraController.shared.setTorchEnabled(enabled, result: result)

        case "detectImage":
          guard let args = call.arguments as? [String: Any],
                let path = args["path"] as? String
          else {
            result(
              FlutterError(
                code: "invalid_args",
                message: "Expected { path: string }",
                details: nil
              )
            )
            return
          }

          guard let image = UIImage(contentsOfFile: path) else {
            result(
              FlutterError(
                code: "invalid_image",
                message: "Could not read image at path.",
                details: nil
              )
            )
            return
          }

          let detections = OnnxCardDetector.shared.detect(image: image)
          result(detections.map { $0.toMap() })

        default:
          result(FlutterMethodNotImplemented)
        }
      }
    }
    GeneratedPluginRegistrant.register(with: self)
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}

private func configureOnnxDetector(registrar: FlutterPluginRegistrar) {
  guard let modelPath = flutterAssetPath(registrar: registrar, asset: "assets/model.onnx"),
        let labelsPath = flutterAssetPath(registrar: registrar, asset: "assets/model_labels.txt")
  else {
    NSLog("[card_detector] model assets not found; ONNX detector disabled.")
    return
  }

  do {
    try OnnxCardDetector.shared.configure(modelPath: modelPath, labelsPath: labelsPath)
    NSLog("[card_detector] ONNX detector configured.")
  } catch {
    NSLog("[card_detector] ONNX configure failed: \(error)")
  }
}

private func flutterAssetPath(registrar: FlutterPluginRegistrar, asset: String) -> String? {
  let key = registrar.lookupKey(forAsset: asset)

  if let path = Bundle.main.path(forResource: key, ofType: nil), FileManager.default.fileExists(atPath: path) {
    return path
  }

  // Flutter assets are commonly stored under App.framework/flutter_assets/.
  if let frameworks = Bundle.main.privateFrameworksURL {
    let appFramework = frameworks.appendingPathComponent("App.framework")
    let candidate = appFramework.appendingPathComponent("flutter_assets").appendingPathComponent(key).path
    if FileManager.default.fileExists(atPath: candidate) { return candidate }
  }

  return nil
}

private final class DetectionEventStreamHandler: NSObject, FlutterStreamHandler {
  static let shared = DetectionEventStreamHandler()

  private var eventSink: FlutterEventSink?

  func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
    eventSink = events
    DetectionBus.shared.setSink(events)
    return nil
  }

  func onCancel(withArguments arguments: Any?) -> FlutterError? {
    eventSink = nil
    DetectionBus.shared.setSink(nil)
    return nil
  }
}

private final class DetectionBus {
  static let shared = DetectionBus()

  private let lock = NSLock()
  private var sink: FlutterEventSink?

  func setSink(_ sink: FlutterEventSink?) {
    lock.lock()
    self.sink = sink
    lock.unlock()
  }

  func emit(_ detections: [[String: Any]]) {
    lock.lock()
    let sink = self.sink
    lock.unlock()

    guard let sink else { return }
    DispatchQueue.main.async { sink(detections) }
  }
}

private final class CardCameraController {
  static let shared = CardCameraController()

  private let lock = NSLock()
  private weak var view: CardCameraPlatformView?

  func attach(_ view: CardCameraPlatformView) {
    lock.lock()
    self.view = view
    lock.unlock()
  }

  func detach(_ view: CardCameraPlatformView) {
    lock.lock()
    if self.view === view { self.view = nil }
    lock.unlock()
  }

  func setTorchEnabled(_ enabled: Bool, result: @escaping FlutterResult) {
    lock.lock()
    let view = self.view
    lock.unlock()

    guard let view else {
      result(false)
      return
    }
    view.setTorchEnabled(enabled, result: result)
  }
}

private final class CardCameraPlatformViewFactory: NSObject, FlutterPlatformViewFactory {
  func createArgsCodec() -> FlutterMessageCodec & NSObjectProtocol {
    FlutterStandardMessageCodec.sharedInstance()
  }

  func create(
    withFrame frame: CGRect,
    viewIdentifier viewId: Int64,
    arguments args: Any?
  ) -> FlutterPlatformView {
    CardCameraPlatformView(frame: frame, viewId: viewId, args: args)
  }
}

private final class PreviewContainerView: UIView {
  let previewLayer: AVCaptureVideoPreviewLayer

  init(previewLayer: AVCaptureVideoPreviewLayer) {
    self.previewLayer = previewLayer
    super.init(frame: .zero)
    layer.addSublayer(previewLayer)
  }

  required init?(coder: NSCoder) { nil }

  override func layoutSubviews() {
    super.layoutSubviews()
    previewLayer.frame = bounds
  }
}

struct Detection {
  let label: String
  let confidence: Double
  let bbox: CGRect

  func toMap() -> [String: Any] {
    [
      "label": label,
      "confidence": confidence,
      "bbox": [
        "x": bbox.origin.x,
        "y": bbox.origin.y,
        "w": bbox.size.width,
        "h": bbox.size.height,
      ],
    ]
  }
}

private final class CardCameraPlatformView: NSObject, FlutterPlatformView, AVCaptureVideoDataOutputSampleBufferDelegate {
  private let viewId: Int64
  private let containerView: UIView
  private let session = AVCaptureSession()
  private let sessionQueue = DispatchQueue(label: "card_detector.camera.session")
  private let videoQueue = DispatchQueue(label: "card_detector.camera.frames")
  private var currentDevice: AVCaptureDevice?
  private var lastInferenceTime: CFTimeInterval = 0
  private let minInferenceInterval: CFTimeInterval = 0.1 // 10 FPS
  private let detector = OnnxCardDetector.shared

  init(frame: CGRect, viewId: Int64, args: Any?) {
    self.viewId = viewId

    let previewLayer = AVCaptureVideoPreviewLayer(session: session)
    previewLayer.videoGravity = .resizeAspectFill
    self.containerView = PreviewContainerView(previewLayer: previewLayer)

    super.init()

    CardCameraController.shared.attach(self)
    configureAndStart()
  }

  deinit {
    CardCameraController.shared.detach(self)
    sessionQueue.async { [session] in
      if session.isRunning { session.stopRunning() }
    }
  }

  func view() -> UIView { containerView }

  private func configureAndStart() {
    sessionQueue.async { [weak self] in
      guard let self else { return }

      self.session.beginConfiguration()
      self.session.sessionPreset = .high

      guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
        self.session.commitConfiguration()
        return
      }
      self.currentDevice = device

      do {
        let input = try AVCaptureDeviceInput(device: device)
        if self.session.canAddInput(input) { self.session.addInput(input) }
      } catch {
        self.session.commitConfiguration()
        return
      }

      let output = AVCaptureVideoDataOutput()
      output.alwaysDiscardsLateVideoFrames = true
      output.videoSettings = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
      ]
      output.setSampleBufferDelegate(self, queue: self.videoQueue)
      if self.session.canAddOutput(output) { self.session.addOutput(output) }

      if let connection = output.connection(with: .video), connection.isVideoOrientationSupported {
        connection.videoOrientation = .portrait
      }

      self.session.commitConfiguration()
      self.session.startRunning()
    }
  }

  func captureOutput(
    _ output: AVCaptureOutput,
    didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    let now = CACurrentMediaTime()
	    if now - lastInferenceTime < minInferenceInterval { return }
	    lastInferenceTime = now
	
	    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
	    var viewSize: CGSize = .zero
	    DispatchQueue.main.sync { [weak self] in
	      guard let self else { return }
	      viewSize = self.containerView.bounds.size
	    }
	    if viewSize.width <= 0 || viewSize.height <= 0 { return }
	
	    let detections = detector.detect(pixelBuffer: pixelBuffer, viewSize: viewSize)
	    DetectionBus.shared.emit(detections.map { $0.toMap() })
	  }

  func setTorchEnabled(_ enabled: Bool, result: @escaping FlutterResult) {
    sessionQueue.async { [weak self] in
      guard let self, let device = self.currentDevice, device.hasTorch else {
        DispatchQueue.main.async { result(false) }
        return
      }
      do {
        try device.lockForConfiguration()
        device.torchMode = enabled ? .on : .off
        device.unlockForConfiguration()
        DispatchQueue.main.async { result(true) }
      } catch {
        DispatchQueue.main.async { result(false) }
      }
    }
  }
}
