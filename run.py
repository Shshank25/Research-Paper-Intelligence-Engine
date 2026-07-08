# ============================================================
# run.py — App Launcher
# Uses the custom python user base installed on E: drive.
# Run with:  python run.py
# ============================================================

import sys
import os
import subprocess

# The custom user base we created to save C: drive space
USER_BASE = r"E:\python_user"

# Update environment to use this user base
env = os.environ.copy()
env["PYTHONUSERBASE"] = USER_BASE

# Find the streamlit executable in the user base Scripts folder
# Under Python 3, user binaries go to Scripts on Windows
streamlit_exe = os.path.join(USER_BASE, "Scripts", "streamlit.exe")

if not os.path.isfile(streamlit_exe):
    # Fallback to module execution via python
    print("Streamlit executable not found. Launching via python module...")
    cmd = [
        sys.executable,
        "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ]
else:
    cmd = [
        streamlit_exe,
        "run", "app.py",
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ]

# Also ensure Python packages from USER_BASE are found
site_pkgs = os.path.join(USER_BASE, "Lib", "site-packages")
if "PYTHONPATH" in env:
    env["PYTHONPATH"] = site_pkgs + ";" + env["PYTHONPATH"]
else:
    env["PYTHONPATH"] = site_pkgs

print("Starting Research Paper Intelligence Engine...")
print(f"   Python  : {sys.executable}")
print(f"   EnvBase : {USER_BASE}")
print(f"   App     : http://localhost:8501\n")

try:
    subprocess.run(cmd, env=env)
except Exception as e:
    print(f"Failed to launch: {e}")
