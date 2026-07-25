@echo off
setlocal

echo Stopping backend (port 8001)...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"

echo Stopping frontend (port 5173)...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"

echo Stopping any leftover flutter/dart processes for this project...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='dart.exe'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*rag_test*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

timeout /t 2 /nobreak >nul

echo Starting backend...
start "RAG Backend" cmd /k "cd /d %~dp0 && env\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001"

timeout /t 3 /nobreak >nul

echo Starting frontend...
start "RAG Frontend" cmd /k "cd /d %~dp0frontend && flutter run -d chrome --web-port=5173"

echo.
echo Backend and frontend are starting in separate windows.
echo Backend:  http://127.0.0.1:8001/api/health
echo Frontend: http://127.0.0.1:5173  (opens automatically in Chrome)
endlocal
