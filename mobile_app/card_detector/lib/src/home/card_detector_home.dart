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

  var _eventCount = 0;
  DateTime? _lastEventAt;

  @override
  void initState() {
    super.initState();
    _controller.detectionsStream.listen((detections) {
      if (!mounted) return;
      setState(() {
        _detections = detections;
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
                          Text('detections: ${_detections.length}'),
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
