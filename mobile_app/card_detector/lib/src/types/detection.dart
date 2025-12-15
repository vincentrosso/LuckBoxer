import 'package:flutter/services.dart';

class Detection {
  const Detection({
    required this.label,
    required this.confidence,
    required this.bbox,
  });

  final String label;
  final double confidence;
  final BBox bbox;

  static List<Detection> listFromEvent(Object? event) {
    if (event is! List) return const [];
    return event
        .whereType<Map>()
        .map((e) => _fromMap(Map<String, Object?>.from(e)))
        .toList(growable: false);
  }

  static Detection _fromMap(Map<String, Object?> map) {
    final bboxMap = map['bbox'];
    if (bboxMap is! Map) {
      throw PlatformException(
        code: 'invalid_detection',
        message: 'Missing bbox map.',
      );
    }

    final bbox = BBox.fromMap(Map<String, Object?>.from(bboxMap));
    return Detection(
      label: (map['label'] as String?) ?? 'card',
      confidence: (map['confidence'] as num?)?.toDouble() ?? 0,
      bbox: bbox,
    );
  }
}

class BBox {
  const BBox({required this.x, required this.y, required this.w, required this.h});

  final double x;
  final double y;
  final double w;
  final double h;

  static BBox fromMap(Map<String, Object?> map) {
    return BBox(
      x: (map['x'] as num).toDouble(),
      y: (map['y'] as num).toDouble(),
      w: (map['w'] as num).toDouble(),
      h: (map['h'] as num).toDouble(),
    );
  }
}
