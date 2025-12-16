import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../types/detection.dart';

const _cameraViewType = 'card_detector/camera_view';
const _detectionsEventChannelName = 'card_detector/detections';
const _cameraControlMethodChannelName = 'card_detector/camera_control';

class CardCameraController {
  CardCameraController();

  final _detectionsController = StreamController<List<Detection>>.broadcast();
  StreamSubscription<dynamic>? _detectionsSub;

  Stream<List<Detection>> get detectionsStream => _detectionsController.stream;

  static const MethodChannel _controlChannel = MethodChannel(
    _cameraControlMethodChannelName,
  );

  void _ensureListening() {
    if (_detectionsSub != null) return;
    const eventChannel = EventChannel(_detectionsEventChannelName);
    _detectionsSub = eventChannel.receiveBroadcastStream().listen(
      (event) {
        final detections = Detection.listFromEvent(event);
        _detectionsController.add(detections);
      },
      onError: (Object error, StackTrace stackTrace) {
        debugPrint('detections stream error: $error');
      },
    );
  }

  Future<List<String>> captureCards({
    int frames = 10,
    int minOccurrences = 3,
    double minConfidence = 0.0,
    Duration timeout = const Duration(seconds: 2),
  }) async {
    _ensureListening();

    final observedLabels = <String>[];
    var seenFrames = 0;
    final done = Completer<void>();

    late final StreamSubscription<List<Detection>> sub;
    sub = detectionsStream.listen((detections) {
      seenFrames += 1;
      for (final d in detections) {
        if (d.confidence >= minConfidence) observedLabels.add(d.label);
      }
      if (seenFrames >= frames && !done.isCompleted) done.complete();
    });

    try {
      await done.future.timeout(timeout);
    } on TimeoutException {
      // Best-effort: return whatever was observed within timeout.
    } finally {
      await sub.cancel();
    }

    final counts = <String, int>{};
    for (final label in observedLabels) {
      counts[label] = (counts[label] ?? 0) + 1;
    }

    final aggregated = counts.entries
        .where((e) => e.value >= minOccurrences)
        .map((e) => e.key)
        .toList(growable: false);
    aggregated.sort();
    return aggregated;
  }

  Future<bool> setTorchEnabled(bool enabled) async {
    try {
      final result = await _controlChannel.invokeMethod<bool>('setTorch', {
        'enabled': enabled,
      });
      return result ?? false;
    } on PlatformException catch (e) {
      debugPrint('setTorch failed: ${e.code} ${e.message}');
      return false;
    }
  }

  Future<List<Detection>> detectImage(String path) async {
    try {
      final result = await _controlChannel.invokeMethod<List>("detectImage", {
        "path": path,
      });
      return Detection.listFromEvent(result);
    } on PlatformException catch (e) {
      debugPrint('detectImage failed: ${e.code} ${e.message}');
      return const [];
    }
  }

  Future<Map<String, Object?>> getOnnxDebug() async {
    try {
      final result = await _controlChannel.invokeMethod<dynamic>('getOnnxDebug');
      if (result is Map) return Map<String, Object?>.from(result);
      return const {};
    } on PlatformException catch (e) {
      debugPrint('getOnnxDebug failed: ${e.code} ${e.message}');
      return const {};
    }
  }

  void dispose() {
    _detectionsSub?.cancel();
    _detectionsSub = null;
    _detectionsController.close();
  }
}

class CardCameraView extends StatefulWidget {
  const CardCameraView({super.key, required this.controller});

  final CardCameraController controller;

  @override
  State<CardCameraView> createState() => _CardCameraViewState();
}

class _CardCameraViewState extends State<CardCameraView> {
  @override
  void initState() {
    super.initState();
    widget.controller._ensureListening();
  }

  @override
  Widget build(BuildContext context) {
    if (defaultTargetPlatform != TargetPlatform.iOS) {
      return const Center(child: Text('iOS only (native camera view).'));
    }

    return const UiKitView(
      viewType: _cameraViewType,
      creationParamsCodec: StandardMessageCodec(),
    );
  }
}
