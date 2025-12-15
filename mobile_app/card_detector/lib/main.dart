import 'package:flutter/material.dart';

import 'src/home/card_detector_home.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CardDetectorApp());
}

class CardDetectorApp extends StatelessWidget {
  const CardDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Card Detector',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal)),
      home: const CardDetectorHome(),
      debugShowCheckedModeBanner: false,
    );
  }
}

