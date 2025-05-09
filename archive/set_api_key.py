import os
import sys

def set_api_key():
    print("Setting up your OpenAI API key...")
    print("Please enter your OpenAI API key (it will be saved to a .env file):")
    api_key = input().strip()
    
    if not api_key:
        print("No API key provided. Exiting.")
        sys.exit(1)
        
    # Save to .env file
    with open('.env', 'w') as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
    
    print("\nAPI key saved to .env file.")
    print("To load the key into your current terminal session, run:")
    print("export OPENAI_API_KEY=your_key_here")
    
    # Also try to set it for the current process
    os.environ["OPENAI_API_KEY"] = api_key
    print("\nAPI key has been set for the current process.")
    print(f"First few characters: {api_key[:4]}...{api_key[-4:]}")
    
    return api_key

if __name__ == "__main__":
    set_api_key()
