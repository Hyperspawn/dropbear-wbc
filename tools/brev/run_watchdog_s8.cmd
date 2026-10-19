@echo off
cd /d H:\dropbear\dropbear-wbc
set DROPBEAR_BREV_SESSION=logs\brev\session_s8.json
"C:\Users\ppare\AppData\Local\Programs\Python\Python312\python.exe" -u tools\brev\watchdog.py >> logs\brev\watchdog_s8_stdout.log 2>&1
