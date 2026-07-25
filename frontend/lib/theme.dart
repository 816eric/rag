import 'package:flutter/material.dart';

class AppColors {
  static const bgMain = Color(0xFF212121);
  static const bgSidebar = Color(0xFF171717);
  static const bgRaised = Color(0xFF2F2F2F);
  static const border = Color(0x1FFFFFFF); // white 12%
  static const text = Color(0xFFECECEC);
  static const textDim = Color(0xFFB4B4B4);
  static const hover = Color(0x14FFFFFF); // white 8%
  static const selected = Color(0x24FFFFFF); // white 14%
}

ThemeData buildAppTheme() {
  return ThemeData(
    brightness: Brightness.dark,
    scaffoldBackgroundColor: AppColors.bgMain,
    colorScheme: const ColorScheme.dark(
      surface: AppColors.bgMain,
      primary: AppColors.text,
      onPrimary: AppColors.bgMain,
    ),
    fontFamily: 'Segoe UI',
    textTheme: const TextTheme(
      bodyMedium: TextStyle(color: AppColors.text),
      bodyLarge: TextStyle(color: AppColors.text),
    ),
    dividerColor: AppColors.border,
  );
}
