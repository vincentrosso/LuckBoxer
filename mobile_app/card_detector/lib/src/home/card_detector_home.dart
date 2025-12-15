import 'package:flutter/material.dart';

import '../platform/card_camera_view.dart';
import '../types/detection.dart';
import '../ui/detection_overlay.dart';

class CardDetectorHome extends StatefulWidget {
  const CardDetectorHome({super.key});

  @override
  State<CardDetectorHome> createState() => _CardDetectorHomeState();
}

class _CardDetectorHomeState extends State<CardDetectorHome> {
  final _controller = CardCameraController();
  var _torchEnabled = false;
  List<Detection> _detections = const [];

  @override
  void initState() {
    super.initState();
    _controller.detectionsStream.listen((detections) {
      if (!mounted) return;
      setState(() => _detections = detections);
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Playing Card Detection'),
        actions: [
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
        ],
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          CardCameraView(controller: _controller),
          IgnorePointer(child: DetectionOverlay(detections: _detections)),
        ],
      ),
    );
  }
}

