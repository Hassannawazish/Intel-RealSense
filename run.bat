@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "DASHBOARD_DIR=%ROOT_DIR%web-dashboard"
set "CONDA_ENV_NAME=face312"
set "CONDA_ACTIVATE_BAT=%USERPROFILE%\anaconda3\Scripts\activate.bat"

echo Starting PPE backend and web dashboard...
echo.

if not exist "%ROOT_DIR%web_dashboard_server.py" (
  echo [ERROR] Backend file not found: "%ROOT_DIR%web_dashboard_server.py"
  pause
  exit /b 1
)

if not exist "%DASHBOARD_DIR%\package.json" (
  echo [ERROR] Frontend package.json not found: "%DASHBOARD_DIR%\package.json"
  pause
  exit /b 1
)

if not exist "%CONDA_ACTIVATE_BAT%" (
  echo [ERROR] Conda activate script not found: "%CONDA_ACTIVATE_BAT%"
  echo Update CONDA_ACTIVATE_BAT in run.bat if your Anaconda is installed elsewhere.
  pause
  exit /b 1
)

start "PPE Backend" cmd /k "call "%CONDA_ACTIVATE_BAT%" %CONDA_ENV_NAME% && cd /d "%ROOT_DIR%" && python .\web_dashboard_server.py"
start "PPE Frontend" cmd /k "cd /d "%DASHBOARD_DIR%" && npm run dev"

echo Backend window started.
echo Frontend window started.
echo.
echo Backend conda env: %CONDA_ENV_NAME%
echo Conda activate:    %CONDA_ACTIVATE_BAT%
echo Backend URL:  http://localhost:8000
echo Frontend URL: http://localhost:5173
echo.
echo Close the opened terminal windows to stop the services.
pause

endlocal
