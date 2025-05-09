import pandas as pd
import os
import subprocess
import sys
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the actual API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")
if api_key and len(api_key) > 10:
    print(f"API key from environment: {api_key[:5]}...{api_key[-4:]}")
    print(f"API key length: {len(api_key)}")
else:
    print("Warning: API key not found or invalid")
    sys.exit(1)

# Hard-coded sample game reviews for demonstration
sample_games = [
    {
        "title": "Black Bull",
        "review": "Black Bull: Slot OverviewOf all the rodeo sports participated in today, bull riding has to be one of the more spectacular. Not many people would want to be in the same pen as one of those nostril-snorting monsters, but brave riders strap themselves on and attempt to last at least 8 seconds without being bucked off. Some of those beasts can buck alright, and it takes a certain kind of person to even think about wanting to ride one. It also takes a certain person to want to fire up Black Bull, an online slot from software developer Pragmatic Play."
    },
    {
        "title": "Hugo 2",
        "review": "With the recent developments within the British gambling industry in mind, we can't help to feel that this game really has the worst possible timing. The UK Gambling Commission, or the UKGC, is the regulator that issues licenses for casinos to operate in the UK, and they recently decided that cartoonish characters are not allowed to be used as a promotion tool."
    },
    {
        "title": "Viper City Heist",
        "review": "Viper City Heist: Slot OverviewA lot of people still cherish the 8-bit aesthetic era even though graphics have moved on leaps and bounds since then. For many gamers raised in the 1980s or early 90s, there's something nostalgic about basic pixelated imagery. The Ninja Turtles, for example, still rock this vibe to this day in certain iterations."
    }
]

# Function to generate a summary using the OpenAI API through curl
def generate_summary(title, review_text):
    """Generate a 4-line summary using OpenAI API via curl"""
    
    # Prepare the API request
    prompt = f"""Please provide a concise 4-line summary of the following casino game review for '{title}':
    
Line 1: Overview focusing on the game theme
Line 2: Focus on the game features
Line 3: Summary of the reviewer's verdict
Line 4: Description of the game aesthetics and audio (if mentioned)

Review: {review_text}"""
    
    # Create a JSON payload for the API request
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a professional game analyst creating concise summaries of casino game reviews."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 300
    }
    
    # Write the payload to a temporary file
    with open('temp_payload.json', 'w') as f:
        json.dump(payload, f)
    
    # Use curl to make the API request
    curl_command = [
        'curl', 'https://api.openai.com/v1/chat/completions',
        '-H', f'Authorization: Bearer {api_key}',
        '-H', 'Content-Type: application/json',
        '-d', '@temp_payload.json'
    ]
    
    try:
        # Execute the curl command
        result = subprocess.run(curl_command, capture_output=True, text=True)
        
        # Check if the request was successful
        if result.returncode == 0:
            # Parse the JSON response
            response = json.loads(result.stdout)
            
            # Extract the summary
            if 'choices' in response and len(response['choices']) > 0:
                summary = response['choices'][0]['message']['content'].strip()
                return summary
            else:
                print(f"Error in API response: {response}")
                return ""
        else:
            print(f"Error executing curl: {result.stderr}")
            return ""
    except Exception as e:
        print(f"Exception during API call: {e}")
        return ""
    finally:
        # Clean up the temporary file
        if os.path.exists('temp_payload.json'):
            os.remove('temp_payload.json')

def process_samples():
    """Process the sample games and generate summaries"""
    
    print(f"Processing {len(sample_games)} sample games...")
    
    # Dictionary to store results
    results = []
    
    # Process each sample game
    for i, game in enumerate(sample_games):
        print(f"\nProcessing game {i+1}/{len(sample_games)}: {game['title']}")
        
        # Generate summary
        summary = generate_summary(game['title'], game['review'])
        
        # Print the summary
        if summary:
            print(f"Summary generated:")
            print(summary)
            
            # Add to results
            results.append({
                "title": game['title'],
                "summary": summary
            })
        else:
            print("Failed to generate summary")
    
    # Save results to a JSON file
    with open('sample_summaries.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummaries saved to sample_summaries.json")
    
    return results

if __name__ == "__main__":
    print("Starting sample summary generation...")
    process_samples()
    print("Done!")
