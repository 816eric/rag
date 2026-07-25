class SessionSummary {
  final String id;
  final String title;

  SessionSummary({required this.id, required this.title});

  factory SessionSummary.fromJson(Map<String, dynamic> json) {
    return SessionSummary(id: json['id'], title: json['title']);
  }
}

class ChatMessage {
  final String role;
  final String content;

  ChatMessage({required this.role, required this.content});

  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    return ChatMessage(role: json['role'], content: json['content']);
  }

  bool get isUser => role == 'user';
}

class AskResult {
  final String answer;
  final double elapsedSeconds;

  AskResult({required this.answer, required this.elapsedSeconds});

  factory AskResult.fromJson(Map<String, dynamic> json) {
    return AskResult(
      answer: json['answer'],
      elapsedSeconds: (json['elapsed_seconds'] as num).toDouble(),
    );
  }
}

class ModelsInfo {
  final String current;
  final List<String> options;

  ModelsInfo({required this.current, required this.options});

  factory ModelsInfo.fromJson(Map<String, dynamic> json) {
    return ModelsInfo(
      current: json['current'],
      options: List<String>.from(json['options']),
    );
  }
}
