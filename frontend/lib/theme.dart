import 'package:flutter/material.dart';

/// Semantic color tokens, following the "name by purpose, not hue" rule -
/// palette values are verified to clear WCAG 2.2 AA (4.5:1 body text,
/// 3:1 large text/UI) rather than eyeballed. Accent is teal, not the
/// indigo #6366f1 that reads as a generic AI-tool default.
class AppColors extends ThemeExtension<AppColors> {
  final Color bg;
  final Color surface;
  final Color surfaceRaised;
  final Color fg;
  final Color muted;
  final Color border;
  final Color accent;
  final Color accentFg;
  final Color success;
  final Color warn;
  final Color danger;

  const AppColors({
    required this.bg,
    required this.surface,
    required this.surfaceRaised,
    required this.fg,
    required this.muted,
    required this.border,
    required this.accent,
    required this.accentFg,
    required this.success,
    required this.warn,
    required this.danger,
  });

  static const dark = AppColors(
    bg: Color(0xFF0F0F0F),
    surface: Color(0xFF1A1A1A),
    surfaceRaised: Color(0xFF242424),
    fg: Color(0xFFF0F0F0),
    muted: Color(0xFFA3A3A3),
    border: Color(0x14FFFFFF), // white 8%
    accent: Color(0xFF2DD4BF), // teal-400, 10.3:1 on bg
    accentFg: Color(0xFF0F0F0F),
    success: Color(0xFF4ADE80),
    warn: Color(0xFFFBBF24),
    danger: Color(0xFFF87171),
  );

  static const light = AppColors(
    bg: Color(0xFFFAFAFA),
    surface: Color(0xFFFFFFFF),
    surfaceRaised: Color(0xFFF4F4F5),
    fg: Color(0xFF111111),
    muted: Color(0xFF5C5C5C), // 6.4:1 on bg
    border: Color(0x14000000), // black 8%
    accent: Color(0xFF0F766E), // teal-700, 5.2:1 text-safe on bg
    accentFg: Color(0xFFFFFFFF),
    success: Color(0xFF15803D), // green-700, 4.8:1
    warn: Color(0xFFB45309), // amber-700, 4.8:1
    danger: Color(0xFFDC2626), // red-600, 4.6:1
  );

  @override
  AppColors copyWith({
    Color? bg,
    Color? surface,
    Color? surfaceRaised,
    Color? fg,
    Color? muted,
    Color? border,
    Color? accent,
    Color? accentFg,
    Color? success,
    Color? warn,
    Color? danger,
  }) {
    return AppColors(
      bg: bg ?? this.bg,
      surface: surface ?? this.surface,
      surfaceRaised: surfaceRaised ?? this.surfaceRaised,
      fg: fg ?? this.fg,
      muted: muted ?? this.muted,
      border: border ?? this.border,
      accent: accent ?? this.accent,
      accentFg: accentFg ?? this.accentFg,
      success: success ?? this.success,
      warn: warn ?? this.warn,
      danger: danger ?? this.danger,
    );
  }

  @override
  AppColors lerp(ThemeExtension<AppColors>? other, double t) {
    if (other is! AppColors) return this;
    return AppColors(
      bg: Color.lerp(bg, other.bg, t)!,
      surface: Color.lerp(surface, other.surface, t)!,
      surfaceRaised: Color.lerp(surfaceRaised, other.surfaceRaised, t)!,
      fg: Color.lerp(fg, other.fg, t)!,
      muted: Color.lerp(muted, other.muted, t)!,
      border: Color.lerp(border, other.border, t)!,
      accent: Color.lerp(accent, other.accent, t)!,
      accentFg: Color.lerp(accentFg, other.accentFg, t)!,
      success: Color.lerp(success, other.success, t)!,
      warn: Color.lerp(warn, other.warn, t)!,
      danger: Color.lerp(danger, other.danger, t)!,
    );
  }
}

extension AppColorsContext on BuildContext {
  AppColors get colors => Theme.of(this).extension<AppColors>()!;
}

ThemeData buildTheme(AppColors colors, Brightness brightness) {
  return ThemeData(
    brightness: brightness,
    scaffoldBackgroundColor: colors.bg,
    colorScheme: ColorScheme(
      brightness: brightness,
      surface: colors.bg,
      onSurface: colors.fg,
      primary: colors.accent,
      onPrimary: colors.accentFg,
      secondary: colors.accent,
      onSecondary: colors.accentFg,
      error: colors.danger,
      onError: colors.accentFg,
    ),
    textTheme: TextTheme(
      bodyMedium: TextStyle(color: colors.fg),
      bodyLarge: TextStyle(color: colors.fg),
    ),
    dividerColor: colors.border,
    extensions: [colors],
  );
}

ThemeData buildDarkTheme() => buildTheme(AppColors.dark, Brightness.dark);
ThemeData buildLightTheme() => buildTheme(AppColors.light, Brightness.light);
