import os
import sys

# Direct read from .env file
def read_api_key_from_file():
    try:
        with open('.env', 'r') as file:
            for line in file:
                if line.strip().startswith('OPENAI_API_KEY='):
                    api_key = line.strip().split('=', 1)[1]
                    # Remove any quotes if present
                    api_key = api_key.strip('\'"')
                    return api_key
    except Exception as e:
        print(f"Error reading .env file: {e}")
    return None

# Get API key from .env file
api_key = read_api_key_from_file()

if api_key and len(api_key) > 10:
    print(f"API key from .env file: {api_key[:5]}...{api_key[-4:]}")
    print(f"API key length: {len(api_key)}")
    
    # Set it in the environment for current process
    os.environ["OPENAI_API_KEY"] = api_key
    print("API key set in environment variable")
else:
    print("No valid API key found in .env file")
    sys.exit(1)

print("\nTesting access to OpenAI API using curl:")
import subprocess

curl_command = [
    'curl', 'https://api.openai.com/v1/models',
    '-H', f'Authorization: Bearer {api_key}',
    '-H', 'Content-Type: application/json'
]

try:
    result = subprocess.run(curl_command, capture_output=True, text=True)
    
    print(f"Exit code: {result.returncode}")
    if result.returncode == 0:
        print("Success! API key is valid.")
        print("First few lines of response:")
        response_lines = result.stdout.split('\n')[:5]
        for line in response_lines:
            print(line[:100] + "..." if len(line) > 100 else line)
    else:
        print("API request failed with error:")
        print(result.stderr)
except Exception as e:
    print(f"Error executing curl: {e}")
