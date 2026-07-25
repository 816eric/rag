import 'package:flutter/material.dart';

import '../models.dart';
import '../theme.dart';

class MessageBubble extends StatelessWidget {
  final ChatMessage message;

  const MessageBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    if (message.isUser) {
      return Align(
        alignment: Alignment.centerRight,
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 6),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          constraints: const BoxConstraints(maxWidth: 560),
          decoration: BoxDecoration(
            color: c.surfaceRaised,
            borderRadius: BorderRadius.circular(20),
          ),
          child: SelectableText(
            message.content,
            style: TextStyle(color: c.fg, fontSize: 15),
          ),
        ),
      );
    }

    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        constraints: const BoxConstraints(maxWidth: 760),
        child: SelectableText(
          message.content,
          style: TextStyle(color: c.fg, fontSize: 15, height: 1.4),
        ),
      ),
    );
  }
}
