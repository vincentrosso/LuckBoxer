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

