@echo off
rem GeoInv3D: start the local API server (unless it is already running) and open the upload page.
rem The server window titled "GeoInv3D API" runs the jobs on this computer (AWS jobs use this
rem machine's AWS access); close it to stop the server and the jobs running here.
setlocal
cd /d "%~dp0"
set PORT=8000
set HEALTH=http://127.0.0.1:%PORT%/api/health
set CHECK=try { if ((Invoke-RestMethod -TimeoutSec 2 %HEALTH%).service -like 'GeoInv3D*') { exit 0 } } catch {}; exit 1

powershell -NoProfile -Command "%CHECK%"
if not errorlevel 1 goto open

echo Starting the GeoInv3D server on port %PORT% ...
start "GeoInv3D API" py -m geoinv3d.api --host 127.0.0.1 --port %PORT%
powershell -NoProfile -Command "for ($i = 0; $i -lt 60; $i++) { try { if ((Invoke-RestMethod -TimeoutSec 2 %HEALTH%).service -like 'GeoInv3D*') { exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }; exit 1"
if errorlevel 1 (
    echo The server did not start within 30 s. See the "GeoInv3D API" window for the error.
    pause
    exit /b 1
)

:open
start "" http://localhost:%PORT%/
endlocal
