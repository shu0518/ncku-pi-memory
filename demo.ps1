chcp 65001 | Out-Null
$env:PYTHON         = "python"
$env:PYTHONPATH     = "."
$env:PI_MEMORY_PATH = "$PWD\memory.json"
$env:PYTHONUTF8     = "1"   
pi -e ./pi-bridge/extension.ts