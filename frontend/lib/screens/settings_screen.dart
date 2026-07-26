import 'dart:html' as html;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../api_client.dart';
import '../theme.dart';

IconData _iconForDoc(String filename) {
  final ext = filename.contains('.') ? filename.split('.').last.toLowerCase() : '';
  switch (ext) {
    case 'pdf':
      return Icons.picture_as_pdf_outlined;
    case 'docx':
      return Icons.description_outlined;
    case 'xlsx':
      return Icons.grid_on_outlined;
    case 'json':
      return Icons.data_object_outlined;
    case 'csv':
      return Icons.table_chart_outlined;
    case 'md':
      return Icons.article_outlined;
    default:
      return Icons.insert_drive_file_outlined;
  }
}

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final ApiClient _api = ApiClient();

  List<String> _documents = [];
  final Set<String> _selected = {}; // starts empty - never pre-select-all,
  // that exact bug once caused "Delete Selected" to wipe out every OTHER document.
  String _searchQuery = '';
  String _folderPath = '';
  bool _includeSubfolders = false;
  String _status = '';
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _refreshDocuments();
  }

  List<String> get _filteredDocuments {
    if (_searchQuery.isEmpty) return _documents;
    final q = _searchQuery.toLowerCase();
    return _documents.where((d) => d.toLowerCase().contains(q)).toList();
  }

  Future<void> _refreshDocuments() async {
    final docs = await _api.listDocuments();
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
    await _api.uploadFiles(result.files);
    await _refreshDocuments();
    setState(() => _busy = false);
  }

  Future<void> _browseFolder() async {
    final path = await _api.browseFolder();
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
    final status = await _api.embedFolder(_folderPath, _includeSubfolders);
    await _refreshDocuments();
    setState(() {
      _status = status;
      _busy = false;
    });
  }

  Future<bool> _confirm(String title, String message, {String confirmLabel = 'Confirm'}) async {
    final c = context.colors;
    final result = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: c.surfaceRaised,
        title: Text(title, style: TextStyle(color: c.fg)),
        content: Text(message, style: TextStyle(color: c.muted)),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: Text('Cancel', style: TextStyle(color: c.muted)),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: Text(confirmLabel, style: TextStyle(color: c.danger, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
    return result ?? false;
  }

  Future<void> _deleteSelected() async {
    if (_selected.isEmpty) return;
    final list = _selected.join(', ');
    final ok = await _confirm(
      'Delete ${_selected.length} document${_selected.length == 1 ? '' : 's'}?',
      'This removes $list from the knowledge base. Files that were uploaded (not folder-referenced) are also deleted from disk. This cannot be undone.',
      confirmLabel: 'Delete',
    );
    if (!ok) return;
    setState(() => _busy = true);
    await _api.deleteDocuments(_selected.toList());
    _selected.clear();
    await _refreshDocuments();
    setState(() => _busy = false);
  }

  Future<void> _restart() async {
    final ok = await _confirm(
      'Restart & clean the entire database?',
      'This wipes ALL embedded documents, not just selected ones. You will need to re-embed everything afterward. This cannot be undone.',
      confirmLabel: 'Wipe everything',
    );
    if (!ok) return;
    setState(() => _busy = true);
    final status = await _api.restart();
    await _refreshDocuments();
    setState(() {
      _status = status;
      _busy = false;
    });
  }

  void _closeTab() {
    html.window.close();
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final filtered = _filteredDocuments;
    final allFilteredSelected = filtered.isNotEmpty && filtered.every(_selected.contains);

    return Scaffold(
      backgroundColor: c.bg,
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 20, 24, 12),
              child: Row(
                children: [
                  Text('📄 Documents & Settings',
                      style: TextStyle(color: c.fg, fontSize: 22, fontWeight: FontWeight.bold)),
                  const Spacer(),
                  IconButton(
                    tooltip: 'Close',
                    onPressed: _closeTab,
                    icon: Icon(Icons.close, color: c.muted),
                  ),
                ],
              ),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 900),
                  child: Wrap(
                    spacing: 16,
                    runSpacing: 16,
                    children: [
                      _buildAddDocumentsCard(c),
                      _buildDocumentListCard(c, filtered, allFilteredSelected),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _card(AppColors c, {required Widget child}) {
    return Container(
      width: 420,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: c.border),
      ),
      child: child,
    );
  }

  Widget _buildAddDocumentsCard(AppColors c) {
    return _card(
      c,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Add documents', style: TextStyle(color: c.fg, fontSize: 16, fontWeight: FontWeight.bold)),
          const SizedBox(height: 14),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _busy ? null : _uploadFiles,
              icon: const Icon(Icons.upload_file, size: 18),
              label: const Text('Upload Files'),
              style: ElevatedButton.styleFrom(backgroundColor: c.accent, foregroundColor: c.accentFg),
            ),
          ),
          const SizedBox(height: 20),
          Text('Embed an entire folder', style: TextStyle(color: c.fg, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
                  decoration: BoxDecoration(
                    color: c.bg,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: c.border),
                  ),
                  child: Text(
                    _folderPath.isEmpty ? 'No folder selected' : _folderPath,
                    style: TextStyle(color: _folderPath.isEmpty ? c.muted : c.fg, fontSize: 13),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              OutlinedButton(
                onPressed: _busy ? null : _browseFolder,
                style: OutlinedButton.styleFrom(foregroundColor: c.fg, side: BorderSide(color: c.border)),
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
          const SizedBox(height: 6),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: _busy ? null : _embedFolder,
              style: ElevatedButton.styleFrom(backgroundColor: c.accent, foregroundColor: c.accentFg),
              child: const Text('Embed Folder'),
            ),
          ),
          if (_status.isNotEmpty) ...[
            const SizedBox(height: 10),
            Text(_status, style: TextStyle(color: c.muted, fontSize: 12)),
          ],
          const SizedBox(height: 24),
          Divider(color: c.border),
          const SizedBox(height: 12),
          Text('Danger zone', style: TextStyle(color: c.danger, fontSize: 13, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          SizedBox(
            width: double.infinity,
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
    );
  }

  Widget _buildDocumentListCard(AppColors c, List<String> filtered, bool allFilteredSelected) {
    return _card(
      c,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text('Embedded Documents', style: TextStyle(color: c.fg, fontSize: 16, fontWeight: FontWeight.bold)),
              const Spacer(),
              Text('${_documents.length} total', style: TextStyle(color: c.muted, fontSize: 12)),
            ],
          ),
          const SizedBox(height: 12),
          TextField(
            style: TextStyle(color: c.fg, fontSize: 13),
            decoration: InputDecoration(
              hintText: 'Search documents...',
              hintStyle: TextStyle(color: c.muted),
              prefixIcon: Icon(Icons.search, color: c.muted, size: 18),
              filled: true,
              fillColor: c.bg,
              contentPadding: const EdgeInsets.symmetric(vertical: 10),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8),
                borderSide: BorderSide(color: c.border),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8),
                borderSide: BorderSide(color: c.border),
              ),
            ),
            onChanged: (v) => setState(() => _searchQuery = v),
          ),
          const SizedBox(height: 8),
          if (filtered.isNotEmpty)
            Row(
              children: [
                Checkbox(
                  value: allFilteredSelected,
                  onChanged: (v) => setState(() {
                    if (v == true) {
                      _selected.addAll(filtered);
                    } else {
                      _selected.removeAll(filtered);
                    }
                  }),
                  activeColor: c.accent,
                  checkColor: c.accentFg,
                ),
                Text('Select all', style: TextStyle(color: c.muted, fontSize: 13)),
                const Spacer(),
                if (_selected.isNotEmpty)
                  Text('${_selected.length} selected', style: TextStyle(color: c.accent, fontSize: 13)),
              ],
            ),
          ConstrainedBox(
            constraints: const BoxConstraints(maxHeight: 320),
            child: filtered.isEmpty
                ? Padding(
                    padding: const EdgeInsets.symmetric(vertical: 24),
                    child: Text(
                      _documents.isEmpty ? 'No documents embedded yet.' : 'No documents match your search.',
                      style: TextStyle(color: c.muted, fontSize: 13),
                    ),
                  )
                : ListView.builder(
                    shrinkWrap: true,
                    itemCount: filtered.length,
                    itemBuilder: (context, index) {
                      final doc = filtered[index];
                      final selected = _selected.contains(doc);
                      return Container(
                        margin: const EdgeInsets.symmetric(vertical: 2),
                        decoration: BoxDecoration(
                          color: selected ? c.accent.withValues(alpha: 0.12) : Colors.transparent,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: CheckboxListTile(
                          value: selected,
                          onChanged: (v) => setState(() {
                            if (v == true) {
                              _selected.add(doc);
                            } else {
                              _selected.remove(doc);
                            }
                          }),
                          secondary: Icon(_iconForDoc(doc), color: c.muted, size: 20),
                          title: Text(doc, style: TextStyle(color: c.fg, fontSize: 13), overflow: TextOverflow.ellipsis),
                          activeColor: c.accent,
                          checkColor: c.accentFg,
                          controlAffinity: ListTileControlAffinity.leading,
                          contentPadding: EdgeInsets.zero,
                          dense: true,
                        ),
                      );
                    },
                  ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: _busy || _selected.isEmpty ? null : _deleteSelected,
              icon: const Icon(Icons.delete_outline, size: 18),
              label: Text('Delete Selected${_selected.isEmpty ? '' : ' (${_selected.length})'}'),
              style: OutlinedButton.styleFrom(
                foregroundColor: c.danger,
                side: BorderSide(color: c.danger.withValues(alpha: 0.5)),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
