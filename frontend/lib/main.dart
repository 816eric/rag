import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'theme.dart';
import 'screens/chat_screen.dart';
import 'screens/settings_screen.dart';

const String _themePrefKey = 'theme_mode';

void main() {
  runApp(const RagApp());
}

class RagApp extends StatefulWidget {
  const RagApp({super.key});

  @override
  State<RagApp> createState() => _RagAppState();
}

class _RagAppState extends State<RagApp> {
  ThemeMode _themeMode = ThemeMode.dark;

  @override
  void initState() {
    super.initState();
    _loadThemePreference();
  }

  Future<void> _loadThemePreference() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString(_themePrefKey);
    if (saved == 'light') {
      setState(() => _themeMode = ThemeMode.light);
    } else if (saved == 'dark') {
      setState(() => _themeMode = ThemeMode.dark);
    }
  }

  Future<void> _toggleTheme() async {
    final next = _themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    setState(() => _themeMode = next);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_themePrefKey, next == ThemeMode.dark ? 'dark' : 'light');
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Local RAG Assistant',
      debugShowCheckedModeBanner: false,
      theme: buildLightTheme(),
      darkTheme: buildDarkTheme(),
      themeMode: _themeMode,
      // No `home:` - `routes` + the default initialRoute (read from the
      // browser's current URL on web) is what lets /#/settings open as a
      // real, independently-addressable route in its own tab.
      routes: {
        '/': (_) => ChatScreen(onToggleTheme: _toggleTheme, themeMode: _themeMode),
        '/settings': (_) => const SettingsScreen(),
      },
    );
  }
}
