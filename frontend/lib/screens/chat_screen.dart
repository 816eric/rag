import 'package:flutter/material.dart';

import '../api_client.dart';
import '../models.dart';
import '../theme.dart';
import '../widgets/message_bubble.dart';
import '../widgets/settings_dialog.dart';

class ChatScreen extends StatefulWidget {
  final VoidCallback onToggleTheme;
  final ThemeMode themeMode;

  const ChatScreen({super.key, required this.onToggleTheme, required this.themeMode});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final ApiClient _api = ApiClient();
  final TextEditingController _questionController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  List<SessionSummary> _sessions = [];
  String? _currentSessionId;
  List<ChatMessage> _messages = [];
  bool _useKnowledge = true;
  bool _asking = false;
  String _elapsed = '';

  ModelsInfo? _modelsInfo;
  String? _selectedModel;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    final sessions = await _api.listSessions();
    final models = await _api.getModels();
    setState(() {
      _sessions = sessions;
      _currentSessionId = sessions.first.id;
      _modelsInfo = models;
      _selectedModel = models.current;
    });
    await _loadMessages(sessions.first.id);
  }

  Future<void> _loadMessages(String sessionId) async {
    final messages = await _api.getMessages(sessionId);
    setState(() {
      _currentSessionId = sessionId;
      _messages = messages;
    });
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
  }

  void _scrollToBottom() {
    if (!_scrollController.hasClients) return;
    _scrollController.animateTo(
      _scrollController.position.maxScrollExtent,
      duration: const Duration(milliseconds: 200),
      curve: Curves.easeOut,
    );
  }

  Future<void> _newChat() async {
    final session = await _api.createSession();
    final sessions = await _api.listSessions();
    setState(() {
      _sessions = sessions;
      _messages = [];
      _currentSessionId = session.id;
    });
  }

  Future<void> _ask() async {
    final question = _questionController.text.trim();
    if (question.isEmpty || _currentSessionId == null) return;
    _questionController.clear();
    setState(() {
      _asking = true;
      _messages = [..._messages, ChatMessage(role: 'user', content: question)];
    });
    // Must wait a frame before scrolling: calling this synchronously right after
    // setState scrolls using the list's PRE-rebuild maxScrollExtent, so the just-sent
    // message never actually comes into view - the chat looks frozen for the entire
    // LLM response time (10-30s) with no sign the question was even received.
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());

    final result = await _api.ask(_currentSessionId!, question, _useKnowledge);
    final sessions = await _api.listSessions();

    setState(() {
      _messages = [..._messages, ChatMessage(role: 'assistant', content: result.answer)];
      _elapsed = '${result.elapsedSeconds.toStringAsFixed(2)} seconds';
      _sessions = sessions;
      _asking = false;
    });
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
  }

  Future<void> _openSettings() async {
    await showDialog(
      context: context,
      builder: (_) => SettingsDialog(api: _api),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Scaffold(
      backgroundColor: c.bg,
      body: Row(
        children: [
          _buildSidebar(c),
          Expanded(child: _buildMain(c)),
        ],
      ),
    );
  }

  Widget _buildSidebar(AppColors c) {
    return Container(
      width: 260,
      color: c.surface,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: _newChat,
                icon: Icon(Icons.add, color: c.fg),
                label: Text('New Chat', style: TextStyle(color: c.fg)),
                style: OutlinedButton.styleFrom(
                  side: BorderSide(color: c.border),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  padding: const EdgeInsets.symmetric(vertical: 12),
                ),
              ),
            ),
          ),
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(horizontal: 8),
              itemCount: _sessions.length,
              itemBuilder: (context, index) {
                final session = _sessions[index];
                final selected = session.id == _currentSessionId;
                return Container(
                  margin: const EdgeInsets.symmetric(vertical: 2),
                  decoration: BoxDecoration(
                    color: selected ? c.accent.withValues(alpha: 0.16) : Colors.transparent,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: ListTile(
                    dense: true,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    title: Text(
                      session.title,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        color: c.fg,
                        fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
                        fontSize: 13,
                      ),
                    ),
                    onTap: () => _loadMessages(session.id),
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(12),
            child: SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: _openSettings,
                icon: Icon(Icons.settings, color: c.fg, size: 18),
                label: Text('Settings', style: TextStyle(color: c.fg)),
                style: OutlinedButton.styleFrom(
                  side: BorderSide(color: c.border),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  padding: const EdgeInsets.symmetric(vertical: 12),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMain(AppColors c) {
    return Column(
      children: [
        _buildHeader(c),
        Expanded(
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 800),
              child: ListView.builder(
                controller: _scrollController,
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                itemCount: _messages.length + (_asking ? 1 : 0),
                itemBuilder: (context, index) {
                  if (index == _messages.length) {
                    return _buildThinkingIndicator(c);
                  }
                  return MessageBubble(message: _messages[index]);
                },
              ),
            ),
          ),
        ),
        _buildInputArea(c),
      ],
    );
  }

  Widget _buildThinkingIndicator(AppColors c) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(strokeWidth: 2, color: c.muted),
            ),
            const SizedBox(width: 10),
            Text('Thinking…', style: TextStyle(color: c.muted, fontSize: 14)),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(AppColors c) {
    final isDark = widget.themeMode == ThemeMode.dark;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      child: Row(
        children: [
          const Text('💬 ', style: TextStyle(fontSize: 20)),
          Text('Local RAG Assistant',
              style: TextStyle(color: c.fg, fontSize: 20, fontWeight: FontWeight.bold)),
          const Spacer(),
          IconButton(
            tooltip: isDark ? 'Switch to light mode' : 'Switch to dark mode',
            onPressed: widget.onToggleTheme,
            icon: Icon(isDark ? Icons.light_mode_outlined : Icons.dark_mode_outlined, color: c.fg),
          ),
          const SizedBox(width: 8),
          if (_modelsInfo != null)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                color: c.surfaceRaised,
                borderRadius: BorderRadius.circular(999),
                border: Border.all(color: c.border),
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: _selectedModel,
                  dropdownColor: c.surfaceRaised,
                  style: TextStyle(color: c.fg, fontSize: 13),
                  items: _modelsInfo!.options
                      .map((m) => DropdownMenuItem(value: m, child: Text(m)))
                      .toList(),
                  onChanged: (value) async {
                    if (value == null) return;
                    await _api.selectModel(value);
                    setState(() => _selectedModel = value);
                  },
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildInputArea(AppColors c) {
    return Padding(
      padding: const EdgeInsets.only(left: 16, right: 16, bottom: 16),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 800),
          child: Column(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                decoration: BoxDecoration(
                  color: c.surfaceRaised,
                  borderRadius: BorderRadius.circular(28),
                  border: Border.all(color: c.border),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _questionController,
                        style: TextStyle(color: c.fg),
                        decoration: InputDecoration(
                          hintText: 'Message the assistant...',
                          hintStyle: TextStyle(color: c.muted),
                          border: InputBorder.none,
                          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                        ),
                        onSubmitted: (_) => _asking ? null : _ask(),
                      ),
                    ),
                    Row(
                      children: [
                        Checkbox(
                          value: _useKnowledge,
                          onChanged: (v) => setState(() => _useKnowledge = v ?? true),
                          activeColor: c.accent,
                          checkColor: c.accentFg,
                        ),
                        Text('Docs', style: TextStyle(color: c.muted, fontSize: 13)),
                      ],
                    ),
                    const SizedBox(width: 4),
                    IconButton(
                      onPressed: _asking ? null : _ask,
                      icon: _asking
                          ? SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2, color: c.accentFg),
                            )
                          : const Icon(Icons.arrow_upward),
                      style: IconButton.styleFrom(
                        backgroundColor: c.accent,
                        foregroundColor: c.accentFg,
                        shape: const CircleBorder(),
                      ),
                    ),
                  ],
                ),
              ),
              if (_elapsed.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(_elapsed, style: TextStyle(color: c.muted, fontSize: 12)),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
