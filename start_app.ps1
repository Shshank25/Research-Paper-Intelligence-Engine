# Auto-generated launcher
$env:PATH = "E:\Miniconda3\Scripts;E:\Miniconda3\envs\rag_env\Scripts;$env:PATH"
Set-Location "E:\VT"
Write-Host "Launching Research Paper Intelligence Engine..." -ForegroundColor Cyan
Write-Host "   URL: http://localhost:8501" -ForegroundColor Green
& "E:\Miniconda3\envs\rag_env\Scripts\streamlit.exe" run "E:\VT\app.py" --server.port 8501 --browser.gatherUsageStats false
