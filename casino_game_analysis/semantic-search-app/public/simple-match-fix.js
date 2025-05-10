// A simplified script to make match percentage text green
// This runs once after page load and after search results appear
// No intervals or mutation observers are used to avoid performance issues

(function() {
  console.log("Simple match percentage fix loaded");
  
  // Function to apply green color to match text
  function applyGreenToMatchText() {
    // Find all paragraphs that might contain match percentage
    const paragraphs = document.querySelectorAll('p.MuiTypography-root, p.MuiTypography-body2');
    
    paragraphs.forEach(p => {
      // Only apply to elements that contain match percentage text
      if (p.textContent && p.textContent.includes('% match')) {
        console.log("Found match percentage text, applying green color");
        p.style.color = '#00ff00';
        p.style.webkitTextFillColor = '#00ff00';
      }
    });
  }
  
  // Run once after initial page load
  setTimeout(applyGreenToMatchText, 1000);
  
  // Set up a one-time event listener for search button clicks
  document.addEventListener('click', function(e) {
    // Check if clicked element is the search button or close to it
    if (e.target.closest('button') || e.target.closest('[role="button"]')) {
      // Wait for search results to appear
      setTimeout(applyGreenToMatchText, 1500);
    }
  });
  
  // Also listen for Enter key on search field
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      // Wait for search results to appear
      setTimeout(applyGreenToMatchText, 1500);
    }
  });
  
  console.log("Simple match percentage fix ready");
})();
