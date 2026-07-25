import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:rag_frontend/main.dart';

void main() {
  testWidgets('App builds without throwing', (WidgetTester tester) async {
    await tester.pumpWidget(const RagApp());
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
