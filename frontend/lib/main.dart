import 'package:flutter/material.dart';

import 'theme.dart';
import 'screens/chat_screen.dart';

void main() {
  runApp(const RagApp());
}

class RagApp extends StatelessWidget {
  const RagApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Local RAG Assistant',
      debugShowCheckedModeBanner: false,
      theme: buildAppTheme(),
      home: const ChatScreen(),
    );
  }
}
