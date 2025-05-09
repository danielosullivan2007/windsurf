// This script directly modifies DOM elements to force specific styling
// It bypasses React's styling system completely

document.addEventListener('DOMContentLoaded', function() {
  // Execute immediately and every second to catch React re-renders
  applyDirectColorFix();
  setInterval(applyDirectColorFix, 1000); 
});

function applyDirectColorFix() {
  console.log('Applying direct DOM color fixes');

  // Look for the subtitle text by its content
  const allParagraphs = document.querySelectorAll('p');
  
  allParagraphs.forEach(p => {
    // Apply dark blue to "Ask for vibes" paragraph
    if (p.textContent.includes('Ask for vibes')) {
      p.style.setProperty('color', '#0033aa', 'important');
      console.log('Applied dark blue to subtitle');
    }
    
    // Apply magenta to "Switch to Standard Search" text
    if (p.textContent.includes('Switch to Standard Search')) {
      p.style.setProperty('color', '#ff4081', 'important');
      console.log('Applied magenta to switch text');
    }
  });
}
