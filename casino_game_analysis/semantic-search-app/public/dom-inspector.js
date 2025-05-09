// DOM Inspector to identify styling issues
document.addEventListener('DOMContentLoaded', function() {
  // Run immediately and after a delay to catch React rendering
  setTimeout(inspectAndFix, 1000);
});

function inspectAndFix() {
  console.log("DOM Inspector running...");
  
  // Find all typography elements
  const paragraphs = document.querySelectorAll('p');
  const headings = document.querySelectorAll('h5');
  
  // Handle headings
  headings.forEach((h5, i) => {
    const text = h5.textContent.trim();
    const computedStyle = window.getComputedStyle(h5);
    
    console.log(`Heading ${i}:`, {
      text: text,
      currentColor: computedStyle.color,
      className: h5.className
    });
    
    if (text.includes('Casino Search')) {
      console.log('Found "Casino Search" heading - applying ice blue');
      h5.style.setProperty('color', '#a0c4ff', 'important');
      
      setTimeout(() => {
        console.log('After heading style application:', window.getComputedStyle(h5).color);
      }, 100);
    }
  });
  
  // Log current styles for paragraphs
  paragraphs.forEach((p, i) => {
    const computedStyle = window.getComputedStyle(p);
    const text = p.textContent.trim().substring(0, 30) + (p.textContent.length > 30 ? '...' : '');
    
    console.log(`Paragraph ${i}:`, {
      text: text,
      currentColor: computedStyle.color,
      fontWeight: computedStyle.fontWeight,
      className: p.className
    });
    
    // Identify specific paragraphs
    if (text.includes('Ask for vibes')) {
      console.log('Found "Ask for vibes" paragraph - applying ice blue');
      p.style.setProperty('color', '#a0c4ff', 'important');
      
      // Debug the style application
      setTimeout(() => {
        console.log('After style application:', window.getComputedStyle(p).color);
      }, 100);
    }
    
    if (text.includes('Switch to Standard')) {
      console.log('Found "Switch to Standard Search" paragraph - applying magenta');
      p.style.setProperty('color', '#ff4081', 'important');
      
      // Debug the style application
      setTimeout(() => {
        console.log('After style application:', window.getComputedStyle(p).color);
      }, 100);
    }
  });
}
