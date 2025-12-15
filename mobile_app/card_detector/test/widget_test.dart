// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter_test/flutter_test.dart';

import 'package:card_detector/main.dart';

void main() {
  testWidgets('Renders home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const CardDetectorApp());
    expect(find.text('Playing Card Detection'), findsOneWidget);
  });
}
