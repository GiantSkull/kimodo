@echo off
rem Copyright Giant Skull: daniel@giantskull.com
rem Double-click to start GS Kimodo (runs the one-time setup when needed).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0gs\launch.ps1"
if errorlevel 1 pause
