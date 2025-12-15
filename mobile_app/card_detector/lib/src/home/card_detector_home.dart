import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../platform/card_camera_view.dart';
import '../types/detection.dart';
import '../ui/detection_overlay.dart';

enum _Mode { camera, photo }

class CardDetectorHome extends StatefulWidget {
  const CardDetectorHome({super.key});

  @override
  State<CardDetectorHome> createState() => _CardDetectorHomeState();
}

class _CardDetectorHomeState extends State<CardDetectorHome> {
  final _controller = CardCameraController();
  final _picker = ImagePicker();

  var _mode = _Mode.camera;
  var _torchEnabled = false;

  List<Detection> _cameraDetections = const [];

  String? _photoPath;
  ui.Size? _photoSize;
  List<Detection> _photoDetections = const [];

  var _eventCount = 0;
  DateTime? _lastEventAt;

  @override
  void initState() {
    super.initState();
    _controller.detectionsStream.listen((detections) {
      if (!mounted) return;
      setState(() {
        _cameraDetections = detections;
        _eventCount += 1;
        _lastEventAt = DateTime.now();
      });
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _pickPhoto() async {
    final image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;

    final bytes = await image.readAsBytes();
    final size = await _decodeImageSize(bytes);

    List<Detection> detections = const [];
    try {
      detections = await _controller.detectImage(image.path);
    } catch (_) {
      detections = const [];
    }

    if (!mounted) return;
    setState(() {
      _mode = _Mode.photo;
      _photoPath = image.path;
      _photoSize = size;
      _photoDetections = detections;
    });
  }

  static Future<ui.Size> _decodeImageSize(Uint8List bytes) async {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, (img) => completer.complete(img));
    final img = await completer.future;
    return ui.Size(img.width.toDouble(), img.height.toDouble());
  }

  @override
  Widget build(BuildContext context) {
    final isCamera = _mode == _Mode.camera;

    return Scaffold(
      appBar: AppBar(
        title: Text(isCamera ? 'Playing Card Detection' : 'Detect From Photo'),
        actions: [
          if (isCamera)
            IconButton(
              tooltip: _torchEnabled ? 'Torch off' : 'Torch on',
              icon: Icon(_torchEnabled ? Icons.flash_on : Icons.flash_off),
              onPressed: () async {
                final next = !_torchEnabled;
                final ok = await _controller.setTorchEnabled(next);
                if (!mounted) return;
                if (ok) setState(() => _torchEnabled = next);
              },
            ),
          IconButton(
            tooltip: 'Pick photo',
            icon: const Icon(Icons.photo_library_outlined),
            onPressed: _pickPhoto,
          ),
          if (!isCamera)
            IconButton(
              tooltip: 'Back to camera',
              icon: const Icon(Icons.videocam_outlined),
              onPressed: () {
                setState(() {
                  _mode = _Mode.camera;
                });
              },
            ),
        ],
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (isCamera) ...[
            CardCameraView(controller: _controller),
            IgnorePointer(child: DetectionOverlay(detections: _cameraDetections)),
          ] else ...[
            _PhotoView(
              path: _photoPath,
              imageSize: _photoSize,
              detectionsNormalized: _photoDetections,
            ),
          ],
          SafeArea(
            child: Align(
              alignment: Alignment.topLeft,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                    child: DefaultTextStyle(
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('events: $_eventCount'),
                          Text('camera det: ${_cameraDetections.length}'),
                          Text('photo det: ${_photoDetections.length}'),
                          Text('last: ${_lastEventAt?.toIso8601String() ?? "—"}'),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _PhotoView extends StatelessWidget {
  const _PhotoView({
    required this.path,
    required this.imageSize,
    required this.detectionsNormalized,
  });

  final String? path;
  final ui.Size? imageSize;
  final List<Detection> detectionsNormalized;

  @override
  Widget build(BuildContext context) {
    if (path == null) {
      return const Center(child: Text('Pick a photo to run detection.'));
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final containerSize = ui.Size(constraints.maxWidth, constraints.maxHeight);
        final intrinsic = imageSize;

        Rect imageRect = Rect.fromLTWH(0, 0, containerSize.width, containerSize.height);
        if (intrinsic != null && intrinsic.width > 0 && intrinsic.height > 0) {
          final scale = _containScale(containerSize, intrinsic);
          final displayW = intrinsic.width * scale;
          final displayH = intrinsic.height * scale;
          final dx = (containerSize.width - displayW) / 2;
          final dy = (containerSize.height - displayH) / 2;
          imageRect = Rect.fromLTWH(dx, dy, displayW, displayH);
        }

        final pixelDetections = detectionsNormalized
            .map((d) => Detection(
                  label: d.label,
                  confidence: d.confidence,
                  bbox: BBox(
                    x: imageRect.left + d.bbox.x * imageRect.width,
                    y: imageRect.top + d.bbox.y * imageRect.height,
                    w: d.bbox.w * imageRect.width,
                    h: d.bbox.h * imageRect.height,
                  ),
                ))
            .toList(growable: false);

        return Stack(
          fit: StackFit.expand,
          children: [
            Center(
              child: FittedBox(
                fit: BoxFit.contain,
                child: Image.file(File(path!)),
              ),
            ),
            IgnorePointer(child: DetectionOverlay(detections: pixelDetections)),
          ],
        );
      },
    );
  }

  static double _containScale(ui.Size container, ui.Size image) {
    final sx = container.width / image.width;
    final sy = container.height / image.height;
    return sx < sy ? sx : sy;
  }
}
