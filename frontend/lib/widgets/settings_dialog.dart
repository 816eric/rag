import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../api_client.dart';
import '../theme.dart';

class SettingsDialog extends StatefulWidget {
  final ApiClient api;

  const SettingsDialog({super.key, required this.api});

  @override
  State<SettingsDialog> createState() => _SettingsDialogState();
}

class _SettingsDialogState extends State<SettingsDialog> {
  List<String> _documents = [];
  final Set<String> _selected = {}; // starts empty - never pre-select-all here,
  // that exact bug once caused "Delete Selected" to wipe out every OTHER document.
  String _folderPath = '';
  bool _includeSubfolders = false;
  String _status = '';
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _refreshDocuments();
  }

  Future<void> _refreshDocuments() async {
    final docs = await widget.api.listDocuments();
    setState(() {
      _documents = docs;
      _selected.removeWhere((d) => !docs.contains(d));
    });
  }

  Future<void> _uploadFiles() async {
    final result = await FilePicker.platform.pickFiles(
      allowMultiple: true,
      withData: true,
      type: FileType.custom,
      allowedExtensions: ['txt', 'pdf', 'docx', 'xlsx', 'md', 'json'],
    );
    if (result == null) return;
    setState(() => _busy = true);
    await widget.api.uploadFiles(result.files);
    await _refreshDocuments();
    setState(() => _busy = false);
  }

  Future<void> _browseFolder() async {
    final path = await widget.api.browseFolder();
    if (path.isNotEmpty) {
      setState(() => _folderPath = path);
    }
  }

  Future<void> _embedFolder() async {
    if (_folderPath.isEmpty) {
      setState(() => _status = 'Please choose a valid folder first.');
      return;
    }
    setState(() => _busy = true);
    final status = await widget.api.embedFolder(_folderPath, _includeSubfolders);
    await _refreshDocuments();
    setState(() {
      _status = status;
      _busy = false;
    });
  }

  Future<void> _deleteSelected() async {
    if (_selected.isEmpty) return;
    setState(() => _busy = true);
    await widget.api.deleteDocuments(_selected.toList());
    _selected.clear();
    await _refreshDocuments();
    setState(() => _busy = false);
  }

  Future<void> _restart() async {
    setState(() => _busy = true);
    final status = await widget.api.restart();
    await _refreshDocuments();
    setState(() {
      _status = status;
      _busy = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Dialog(
      backgroundColor: c.surfaceRaised,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 440, maxHeight: 640),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text('Documents',
                        style: TextStyle(color: c.fg, fontSize: 18, fontWeight: FontWeight.bold)),
                  ),
                  IconButton(
                    icon: Icon(Icons.close, color: c.fg),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      ElevatedButton.icon(
                        onPressed: _busy ? null : _uploadFiles,
                        icon: const Icon(Icons.upload_file),
                        label: const Text('Upload Files'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: c.accent,
                          foregroundColor: c.accentFg,
                        ),
                      ),
                      const SizedBox(height: 16),
                      Text('Embed an entire folder',
                          style: TextStyle(color: c.fg, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 8),
                      Row(
                        children: [
                          Expanded(
                            child: Container(
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                              decoration: BoxDecoration(
                                color: c.bg,
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(color: c.border),
                              ),
                              child: Text(
                                _folderPath.isEmpty ? 'No folder selected' : _folderPath,
                                style: TextStyle(
                                    color: _folderPath.isEmpty ? c.muted : c.fg, fontSize: 13),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          OutlinedButton(
                            onPressed: _busy ? null : _browseFolder,
                            style: OutlinedButton.styleFrom(
                              foregroundColor: c.fg,
                              side: BorderSide(color: c.border),
                            ),
                            child: const Text('Browse...'),
                          ),
                        ],
                      ),
                      CheckboxListTile(
                        value: _includeSubfolders,
                        onChanged: (v) => setState(() => _includeSubfolders = v ?? false),
                        title: Text('Include subfolders', style: TextStyle(color: c.fg)),
                        activeColor: c.accent,
                        checkColor: c.accentFg,
                        controlAffinity: ListTileControlAffinity.leading,
                        contentPadding: EdgeInsets.zero,
                        dense: true,
                      ),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: _busy ? null : _embedFolder,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: c.accent,
                            foregroundColor: c.accentFg,
                          ),
                          child: const Text('Embed Folder'),
                        ),
                      ),
                      if (_status.isNotEmpty) ...[
                        const SizedBox(height: 8),
                        Text(_status, style: TextStyle(color: c.muted, fontSize: 12)),
                      ],
                      const SizedBox(height: 16),
                      Text('Embedded Documents',
                          style: TextStyle(color: c.fg, fontWeight: FontWeight.bold)),
                      ..._documents.map((doc) => CheckboxListTile(
                            value: _selected.contains(doc),
                            onChanged: (v) => setState(() {
                              if (v == true) {
                                _selected.add(doc);
                              } else {
                                _selected.remove(doc);
                              }
                            }),
                            title: Text(doc, style: TextStyle(color: c.fg, fontSize: 13)),
                            activeColor: c.accent,
                            checkColor: c.accentFg,
                            controlAffinity: ListTileControlAffinity.leading,
                            contentPadding: EdgeInsets.zero,
                            dense: true,
                          )),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: OutlinedButton(
                              onPressed: _busy || _selected.isEmpty ? null : _deleteSelected,
                              style: OutlinedButton.styleFrom(
                                foregroundColor: c.danger,
                                side: BorderSide(color: c.danger.withValues(alpha: 0.5)),
                              ),
                              child: const Text('Delete Selected'),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: OutlinedButton(
                              onPressed: _busy ? null : _restart,
                              style: OutlinedButton.styleFrom(
                                foregroundColor: c.danger,
                                side: BorderSide(color: c.danger.withValues(alpha: 0.5)),
                              ),
                              child: const Text('Restart & Clean DB'),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
