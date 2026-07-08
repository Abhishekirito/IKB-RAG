import os
import requests
from dotenv import load_dotenv

load_dotenv()
url = 'https://abhiswork-mineru-api-ikb.hf.space/file_parse'
token = os.getenv('HF_TOKEN')
print(f"Token present: {bool(token)}")

headers = {'Authorization': f'Bearer {token}'}

try:
    with open('uploaded_documents/462-Piping-and-Instrumentation-Diagrams.pdf', 'rb') as f:
        files = {'files': ('462-Piping-and-Instrumentation-Diagrams.pdf', f)}
        print("Sending request...")
        data = {'backend': 'pipeline', 'is_ocr': False}
        res = requests.post(url, files=files, data=data, headers=headers)
    print(f"Status: {res.status_code}")
    import json
    data = res.json()
    print("Keys:", data.keys())
    if "data" in data:
        print("Data Keys:", data["data"].keys())
        if "markdown" in data["data"]:
            print("Markdown sample:", data["data"]["markdown"][:500])
except Exception as e:
    print(f"Exception: {e}")
