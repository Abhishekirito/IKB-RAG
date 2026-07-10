import os
import subprocess
import time
import requests

try:
    os.system("powershell -Command \"Get-Process -Id (Get-NetTCPConnection -LocalPort 8002).OwningProcess | Stop-Process -Force\"")
except:
    pass

p = subprocess.Popen(["python", "api.py"])
print("Waiting for server to start...")
time.sleep(6)

try:
    res = requests.post("http://127.0.0.1:8002/query", json={"question": "test", "chat_id": "1783699086638"})
    print("Status:", res.status_code)
    print("Response:", res.text)
except Exception as e:
    print("Request failed:", e)

p.kill()
