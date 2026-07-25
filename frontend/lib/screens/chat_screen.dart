import 'package:flutter/material.dart';

import '../api_client.dart';
import '../models.dart';
import '../theme.dart';
import '../widgets/message_bubble.dart';
import '../widgets/settings_dialog.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

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
    _scrollToBottom();

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
    return Scaffold(
      backgroundColor: AppColors.bgMain,
      body: Row(
        children: [
          _buildSidebar(),
          Expanded(child: _buildMain()),
        ],
      ),
    );
  }

  Widget _buildSidebar() {
    return Container(
      width: 260,
      color: AppColors.bgSidebar,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: _newChat,
                icon: const Icon(Icons.add, color: AppColors.text),
                label: const Text('New Chat', style: TextStyle(color: AppColors.text)),
                style: OutlinedButton.styleFrom(
                  side: const BorderSide(color: AppColors.border),
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
                    color: selected ? AppColors.selected : Colors.transparent,
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
                        color: AppColors.text,
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
                icon: const Icon(Icons.settings, color: AppColors.text, size: 18),
                label: const Text('Settings', style: TextStyle(color: AppColors.text)),
                style: OutlinedButton.styleFrom(
                  side: const BorderSide(color: AppColors.border),
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

  Widget _buildMain() {
    return Column(
      children: [
        _buildHeader(),
        Expanded(
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 800),
              child: ListView.builder(
                controller: _scrollController,
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                itemCount: _messages.length,
                itemBuilder: (context, index) => MessageBubble(message: _messages[index]),
              ),
            ),
          ),
        ),
        _buildInputArea(),
      ],
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      child: Row(
        children: [
          const Text('💬 ', style: TextStyle(fontSize: 20)),
          const Text('Local RAG Assistant',
              style: TextStyle(color: AppColors.text, fontSize: 20, fontWeight: FontWeight.bold)),
          const Spacer(),
          if (_modelsInfo != null)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                color: AppColors.bgRaised,
                borderRadius: BorderRadius.circular(999),
                border: Border.all(color: AppColors.border),
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: _selectedModel,
                  dropdownColor: AppColors.bgRaised,
                  style: const TextStyle(color: AppColors.text, fontSize: 13),
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

  Widget _buildInputArea() {
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
                  color: AppColors.bgRaised,
                  borderRadius: BorderRadius.circular(28),
                  border: Border.all(color: AppColors.border),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _questionController,
                        style: const TextStyle(color: AppColors.text),
                        decoration: const InputDecoration(
                          hintText: 'Message the assistant...',
                          hintStyle: TextStyle(color: AppColors.textDim),
                          border: InputBorder.none,
                          contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                        ),
                        onSubmitted: (_) => _asking ? null : _ask(),
                      ),
                    ),
                    Row(
                      children: [
                        Checkbox(
                          value: _useKnowledge,
                          onChanged: (v) => setState(() => _useKnowledge = v ?? true),
                          activeColor: AppColors.text,
                          checkColor: AppColors.bgMain,
                        ),
                        const Text('Docs', style: TextStyle(color: AppColors.textDim, fontSize: 13)),
                      ],
                    ),
                    const SizedBox(width: 4),
                    IconButton(
                      onPressed: _asking ? null : _ask,
                      icon: _asking
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2, color: AppColors.text),
                            )
                          : const Icon(Icons.arrow_upward),
                      style: IconButton.styleFrom(
                        backgroundColor: AppColors.text,
                        foregroundColor: AppColors.bgMain,
                        shape: const CircleBorder(),
                      ),
                    ),
                  ],
                ),
              ),
              if (_elapsed.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(_elapsed, style: const TextStyle(color: AppColors.textDim, fontSize: 12)),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
