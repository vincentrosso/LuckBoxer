import 'package:flutter/material.dart';

import '../types/detection.dart';

class DetectionOverlay extends StatelessWidget {
  const DetectionOverlay({super.key, required this.detections});

  final List<Detection> detections;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(painter: _DetectionPainter(detections));
  }
}

class _DetectionPainter extends CustomPainter {
  _DetectionPainter(this.detections);

  final List<Detection> detections;

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.lightGreenAccent;

    for (final d in detections) {
      final rect = Rect.fromLTWH(d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h);
      canvas.drawRect(rect, boxPaint);

      final label = '${d.label} ${(d.confidence * 100).toStringAsFixed(0)}%';
      final textSpan = TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.black,
          fontSize: 12,
          fontWeight: FontWeight.w600,
        ),
      );
      final tp = TextPainter(text: textSpan, textDirection: TextDirection.ltr)
        ..layout(maxWidth: size.width);

      final padding = const EdgeInsets.symmetric(horizontal: 6, vertical: 3);
      final bgRect = Rect.fromLTWH(
        rect.left,
        (rect.top - tp.height - padding.vertical).clamp(0, size.height),
        tp.width + padding.horizontal,
        tp.height + padding.vertical,
      );

      canvas.drawRRect(
        RRect.fromRectAndRadius(bgRect, const Radius.circular(6)),
        Paint()..color = Colors.lightGreenAccent,
      );
      tp.paint(canvas, Offset(bgRect.left + padding.left, bgRect.top + padding.top));
    }
  }

  @override
  bool shouldRepaint(_DetectionPainter oldDelegate) =>
      oldDelegate.detections != detections;
}

