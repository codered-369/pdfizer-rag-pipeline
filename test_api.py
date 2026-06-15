import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key or api_key == "your_google_api_key_here":
    print("API key not found in .env file.")
    exit(1)

print("Checking API key...")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
try:
    response = requests.get(url)

    if response.status_code == 200:
        print("SUCCESS! Your Google API Key is valid.")
        data = response.json()
        print("\n=== ALL Available Models ===")
        for m in data.get("models", []):
            name = m.get('name')
            methods = ", ".join(m.get('supportedGenerationMethods', []))
            print(f" - {name} (Supports: {methods})")
    else:
        print(f"FAILED! The API key is invalid or an error occurred. Status Code: {response.status_code}")
        print(f"Error Details: {response.text}")
except Exception as e:
    print(f"FAILED! Could not reach the Google API. Error: {e}")
