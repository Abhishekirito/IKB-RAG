from gradio_client import Client
import json
import time

space_id = "abhiswork/IKB-nomic"
try:
    print(f"Connecting to {space_id}...")
    client = Client(space_id)
    
    texts = ["Test sentence 1", "Test sentence 2", "Test sentence 3", "Test sentence 4"]
    print(f"Sending {len(texts)} texts...")
    
    start_time = time.time()
    result_json = client.predict(json.dumps(texts), api_name="/generate_embeddings")
    end_time = time.time()
    
    result = json.loads(result_json)
    if isinstance(result, dict) and "error" in result:
        print(f"Space returned error: {result['error']}")
    else:
        print(f"Success! Received {len(result)} embeddings in {end_time - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error connecting or predicting: {e}")
