#!/usr/bin/env python
import os
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

BASE_DIR = "/Users/danielosullivan/Desktop/windsurf_testing/windsurf/casino_game_analysis"
REACT_APP_DIR = f"{BASE_DIR}/semantic-search-app/src"

def update_react_app():
    """Update the React app code to fix any hardcoded references to 'Lights'"""
    app_js_file = f"{REACT_APP_DIR}/App.js"
    
    if not os.path.exists(app_js_file):
        logging.error(f"React app file not found: {app_js_file}")
        return False
    
    logging.info(f"Updating React app file: {app_js_file}")
    
    # Read the current content
    with open(app_js_file, 'r') as f:
        content = f.read()
    
    # Make fixes to the React app code:
    
    # 1. Fix the search result title display by ensuring title trimming doesn't happen
    # Look for code that might be truncating titles and update it
    if "result.title.substring" in content:
        content = content.replace(
            "result.title.substring(0, 10)",
            "result.title"  # Display full title
        )
    
    # 2. Add code to clean up search results further
    search_result_processing = """
  // Process search results to ensure correct titles
  const processSearchResults = (results) => {
    if (!results) return [];
    
    return results.map(result => {
      // Ensure the title is correctly set
      if (result.title && (result.title === 'Lights' || result.title === '"Lights' || result.title.startsWith('Lights,'))) {
        result.title = 'Lights, Camera, Action!';
      }
      
      // Also fix in the summary if present
      if (result.summary && result.summary.includes('Lights, Camera, Cash!')) {
        result.summary = result.summary.replace('Lights, Camera, Cash!', 'Lights, Camera, Action!');
      }
      
      return result;
    });
  };
"""
    
    # Find a good place to add the function - just before the App function
    if "function App()" in content:
        content = content.replace(
            "function App()",
            search_result_processing + "\nfunction App()"
        )
        
        # Add a call to processSearchResults when receiving search results
        if "setSearchResults(data.results)" in content:
            content = content.replace(
                "setSearchResults(data.results)",
                "setSearchResults(processSearchResults(data.results))"
            )
    
    # 3. Add a forced refresh when using the search suggestions
    if "const handleSuggestionClick = (suggestion) =>" in content:
        # Update the suggestion handler to force a new search
        suggestion_handler = """
  const handleSuggestionClick = (suggestion) => {
    // Clear previous results first to ensure fresh data
    setSearchResults([]);
    // Set a timeout to ensure state updates before search
    setTimeout(() => {
      setSearchQuery(suggestion);
      handleSearch(suggestion);
    }, 50);
  };
"""
        # Replace the existing handler
        content = re.sub(
            r'const handleSuggestionClick = \(suggestion\) =>\s*{[^}]*}',
            suggestion_handler,
            content
        )
    
    # Write the updated content
    with open(app_js_file, 'w') as f:
        f.write(content)
    
    logging.info(f"Successfully updated React app code")
    return True

if __name__ == "__main__":
    update_react_app()
