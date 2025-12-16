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
  var _snapshotBusy = false;

  List<Detection> _cameraDetections = const [];

  Map<String, Object?> _onnxDebug = const {};
  Timer? _onnxTimer;

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

    _onnxTimer = Timer.periodic(const Duration(seconds: 1), (_) async {
      final info = await _controller.getOnnxDebug();
      if (!mounted) return;
      setState(() => _onnxDebug = info);
    });
  }

  @override
  void dispose() {
    _onnxTimer?.cancel();
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

  Future<void> _captureCardsSnapshot() async {
    if (_snapshotBusy) return;
    setState(() => _snapshotBusy = true);

    try {
      final labels = await _controller.captureCards(
        frames: 10,
        minOccurrences: 3,
        minConfidence: 0.0,
        timeout: const Duration(seconds: 2),
      );

      final sorted = _sortCardLabelsAllTrumps(labels);

      if (!mounted) return;
      await showModalBottomSheet<void>(
        context: context,
        showDragHandle: true,
        builder: (context) {
          if (sorted.isEmpty) {
            return const Padding(
              padding: EdgeInsets.all(16),
              child: Text('No stable cards detected (need ≥3 hits across ~10 frames).'),
            );
          }
          return ListView.separated(
            padding: const EdgeInsets.all(16),
            itemCount: sorted.length,
            separatorBuilder: (context, index) => const Divider(height: 1),
            itemBuilder: (context, i) {
              final label = sorted[i];
              return ListTile(
                title: Text(label),
                subtitle: const Text('Snapshot (aggregated over multiple frames)'),
              );
            },
          );
        },
      );
    } finally {
      if (mounted) setState(() => _snapshotBusy = false);
    }
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
          if (isCamera)
            IconButton(
              tooltip: 'Snapshot (aggregate)',
              icon: _snapshotBusy
                  ? const SizedBox.square(
                      dimension: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.center_focus_strong_outlined),
              onPressed: _snapshotBusy ? null : _captureCardsSnapshot,
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
                          Text('onnx: ${_onnxDebug['configured'] ?? '—'}'),
                          Text('shape: ${_onnxDebug['outputShape'] ?? '—'}'),
                          Text('max: ${_onnxDebug['maxLabel'] ?? ''} ${_onnxDebug['maxScore'] ?? ''}'),
                          Text('cand/sel: ${_onnxDebug['candidates'] ?? '—'}/${_onnxDebug['selected'] ?? '—'}'),
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

List<String> _sortCardLabelsAllTrumps(List<String> labels) {
  final parsed = labels.map(_parseLabel).whereType<_ParsedCard>().toList(growable: false);
  parsed.sort((a, b) {
    final s = a.suitIndex.compareTo(b.suitIndex);
    if (s != 0) return s;
    return a.orderIndex.compareTo(b.orderIndex);
  });
  return parsed.map((c) => c.label).toList(growable: false);
}

_ParsedCard? _parseLabel(String label) {
  if (label.length < 2) return null;
  final suit = label.substring(label.length - 1).toUpperCase();
  final value = label.substring(0, label.length - 1).toUpperCase();

  final suitIndex = switch (suit) {
    'S' => 0,
    'H' => 1,
    'D' => 2,
    'C' => 3,
    _ => null,
  };
  if (suitIndex == null) return null;

  final orderIndex = _allTrumpsOrderIndex(value);
  if (orderIndex == null) return null;

  return _ParsedCard(label: label, suitIndex: suitIndex, orderIndex: orderIndex);
}

int? _allTrumpsOrderIndex(String value) {
  // Matches Playing-Cards-Object-Detection/demo_application/utils/game_logic.py CardTrumpOrder
  return switch (value) {
    'J' => 0,
    '9' => 1,
    'A' => 2,
    '10' => 3,
    'K' => 4,
    'Q' => 5,
    '8' => 6,
    '7' => 7,
    _ => null,
  };
}

class _ParsedCard {
  const _ParsedCard({
    required this.label,
    required this.suitIndex,
    required this.orderIndex,
  });

  final String label;
  final int suitIndex;
  final int orderIndex;
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
