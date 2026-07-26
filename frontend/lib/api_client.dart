import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';

import 'models.dart';

class ApiClient {
  static const String baseUrl = 'http://127.0.0.1:8001';

  Future<List<SessionSummary>> listSessions() async {
    final res = await http.get(Uri.parse('$baseUrl/api/sessions'));
    final List<dynamic> data = jsonDecode(res.body);
    return data.map((e) => SessionSummary.fromJson(e)).toList();
  }

  Future<SessionSummary> createSession() async {
    final res = await http.post(Uri.parse('$baseUrl/api/sessions'));
    return SessionSummary.fromJson(jsonDecode(res.body));
  }

  Future<List<ChatMessage>> getMessages(String sessionId) async {
    final res = await http.get(Uri.parse('$baseUrl/api/sessions/$sessionId/messages'));
    final List<dynamic> data = jsonDecode(res.body);
    return data.map((e) => ChatMessage.fromJson(e)).toList();
  }

  Future<AskResult> ask(String sessionId, String question, bool useKnowledge) async {
    final res = await http.post(
      Uri.parse('$baseUrl/api/sessions/$sessionId/ask'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'question': question, 'use_knowledge': useKnowledge}),
    );
    return AskResult.fromJson(jsonDecode(res.body));
  }

  Future<List<String>> listDocuments() async {
    final res = await http.get(Uri.parse('$baseUrl/api/documents'));
    final List<dynamic> data = jsonDecode(res.body);
    return data.cast<String>();
  }

  Future<List<String>> uploadFiles(List<PlatformFile> files) async {
    final request = http.MultipartRequest('POST', Uri.parse('$baseUrl/api/documents/upload'));
    for (final f in files) {
      request.files.add(http.MultipartFile.fromBytes('files', f.bytes!, filename: f.name));
    }
    final streamed = await request.send();
    final res = await http.Response.fromStream(streamed);
    final List<dynamic> data = jsonDecode(res.body);
    return data.cast<String>();
  }

  Future<String> browseFolder() async {
    final res = await http.post(Uri.parse('$baseUrl/api/documents/browse'));
    return jsonDecode(res.body)['folder_path'];
  }

  Future<String> embedFolder(String folderPath, bool includeSubfolders) async {
    final res = await http.post(
      Uri.parse('$baseUrl/api/documents/folder'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'folder_path': folderPath, 'include_subfolders': includeSubfolders}),
    );
    return jsonDecode(res.body)['status'];
  }

  Future<List<String>> deleteDocuments(List<String> names) async {
    final res = await http.post(
      Uri.parse('$baseUrl/api/documents/delete'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'names': names}),
    );
    final List<dynamic> data = jsonDecode(res.body);
    return data.cast<String>();
  }

  Future<ModelsInfo> getModels() async {
    final res = await http.get(Uri.parse('$baseUrl/api/models'));
    return ModelsInfo.fromJson(jsonDecode(res.body));
  }

  Future<String> selectModel(String modelName) async {
    final res = await http.post(
      Uri.parse('$baseUrl/api/models'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'model_name': modelName}),
    );
    return jsonDecode(res.body)['status'];
  }

  Future<String> restart() async {
    final res = await http.post(Uri.parse('$baseUrl/api/system/restart'));
    return jsonDecode(res.body)['status'];
  }

  Future<String> transcribeAudio(Uint8List audioBytes, String filename) async {
    final request = http.MultipartRequest('POST', Uri.parse('$baseUrl/api/voice/transcribe'));
    request.files.add(http.MultipartFile.fromBytes('file', audioBytes, filename: filename));
    final streamed = await request.send();
    final res = await http.Response.fromStream(streamed);
    return jsonDecode(res.body)['text'];
  }

  /// Returns WAV audio bytes, or null if the backend had nothing to say (empty text).
  Future<Uint8List?> synthesizeSpeech(String text) async {
    final res = await http.post(
      Uri.parse('$baseUrl/api/voice/synthesize'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'text': text}),
    );
    if (res.statusCode == 204 || res.bodyBytes.isEmpty) return null;
    return res.bodyBytes;
  }

  /// Streams the answer as it's generated. Each event is the raw decoded JSON
  /// from one SSE `data:` line: `{'delta': '...'}` while generating, then a
  /// final `{'done': true, 'elapsed_seconds': ...}`.
  Stream<Map<String, dynamic>> askStream(String sessionId, String question, bool useKnowledge) async* {
    final request = http.Request('POST', Uri.parse('$baseUrl/api/sessions/$sessionId/ask_stream'))
      ..headers['Content-Type'] = 'application/json'
      ..body = jsonEncode({'question': question, 'use_knowledge': useKnowledge});

    final streamedResponse = await http.Client().send(request);
    final lines = streamedResponse.stream.transform(utf8.decoder).transform(const LineSplitter());
    await for (final line in lines) {
      if (!line.startsWith('data: ')) continue;
      yield jsonDecode(line.substring(6)) as Map<String, dynamic>;
    }
  }
}
