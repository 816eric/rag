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
    return Dialog(
      backgroundColor: AppColors.bgRaised,
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
                  const Expanded(
                    child: Text('Documents',
                        style: TextStyle(color: AppColors.text, fontSize: 18, fontWeight: FontWeight.bold)),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: AppColors.text),
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
                      ),
                      const SizedBox(height: 16),
                      const Text('Embed an entire folder',
                          style: TextStyle(color: AppColors.text, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 8),
                      Row(
                        children: [
                          Expanded(
                            child: Container(
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                              decoration: BoxDecoration(
                                color: AppColors.bgMain,
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(color: AppColors.border),
                              ),
                              child: Text(
                                _folderPath.isEmpty ? 'No folder selected' : _folderPath,
                                style: TextStyle(
                                    color: _folderPath.isEmpty ? AppColors.textDim : AppColors.text, fontSize: 13),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          OutlinedButton(onPressed: _busy ? null : _browseFolder, child: const Text('Browse...')),
                        ],
                      ),
                      CheckboxListTile(
                        value: _includeSubfolders,
                        onChanged: (v) => setState(() => _includeSubfolders = v ?? false),
                        title: const Text('Include subfolders', style: TextStyle(color: AppColors.text)),
                        controlAffinity: ListTileControlAffinity.leading,
                        contentPadding: EdgeInsets.zero,
                        dense: true,
                      ),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(onPressed: _busy ? null : _embedFolder, child: const Text('Embed Folder')),
                      ),
                      if (_status.isNotEmpty) ...[
                        const SizedBox(height: 8),
                        Text(_status, style: const TextStyle(color: AppColors.textDim, fontSize: 12)),
                      ],
                      const SizedBox(height: 16),
                      const Text('Embedded Documents',
                          style: TextStyle(color: AppColors.text, fontWeight: FontWeight.bold)),
                      ..._documents.map((doc) => CheckboxListTile(
                            value: _selected.contains(doc),
                            onChanged: (v) => setState(() {
                              if (v == true) {
                                _selected.add(doc);
                              } else {
                                _selected.remove(doc);
                              }
                            }),
                            title: Text(doc, style: const TextStyle(color: AppColors.text, fontSize: 13)),
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
                              child: const Text('Delete Selected'),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: OutlinedButton(
                              onPressed: _busy ? null : _restart,
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
