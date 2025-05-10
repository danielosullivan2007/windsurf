// This script specifically targets and forces the match percentage text to be vibrant green
// It runs on an interval to catch any dynamic content changes with multiple fallback approaches

/* TEMPORARILY DISABLED - causing search performance issues */
/*
(function() {
  // Create a global style rule as a first line of defense
  const initialStyle = document.createElement('style');
  initialStyle.innerHTML = `
    /* All possible match percentage text selectors */
    p.MuiTypography-root.MuiTypography-body2[data-component-name="GreenMatchText-root"],
    .GreenMatchText-root,
    p:has(text):contains("% match"),
    p.MuiTypography-root:has(text):contains("% match"),
    .MuiTypography-body2:contains("% match"),
    body .MuiTypography-root:contains("% match") {
      color: #00ff00 !important;
      -webkit-text-fill-color: #00ff00 !important;
    }
  `;
  document.head.appendChild(initialStyle);
  
  let fixAttempts = 0;
  
  function fixMatchPercentageColor() {
    fixAttempts++;
    console.log(`Running match percentage color fix... (attempt ${fixAttempts})`);
    
    // Find all possible containers of match text
    const paragraphs = Array.from(document.querySelectorAll('p, .MuiTypography-root, .MuiTypography-body2'));
    
    paragraphs.forEach(p => {
      if (p.textContent.includes('% match')) {
        console.log("FOUND match percentage text:", p.textContent);
        
        // Force style with multiple approaches for maximum reliability
        // This brute force approach should override any Material UI styling
        
        // 1. Direct inline style with !important (highest CSS priority)
        p.style.cssText = 'color: #00ff00 !important; -webkit-text-fill-color: #00ff00 !important;';
        
        // 2. Add a unique ID and create a style rule for it
        const elementId = 'match-text-' + fixAttempts + '-' + Math.random().toString(36).substr(2, 5);
        p.id = elementId;
        
        const styleEl = document.createElement('style');
        styleEl.innerHTML = `
          /* Triple ID selector for extreme specificity */
          #${elementId}#${elementId}#${elementId} {
            color: #00ff00 !important;
            -webkit-text-fill-color: #00ff00 !important;
          }
        `;
        document.head.appendChild(styleEl);
        
        // 3. Desperate measure: replace the text with a span
        const textContent = p.textContent;
        if (textContent.includes('% match')) {
          p.innerHTML = textContent.replace(
            /(\d+)% match/g, 
            '<span class="forced-green-text" style="color: #00ff00 !important; -webkit-text-fill-color: #00ff00 !important;">$1% match</span>'
          );
        }
        
        // Log success/failure
        setTimeout(() => {
          const computedColor = window.getComputedStyle(p).color;
          console.log('Current color:', computedColor);
          if (computedColor !== 'rgb(0, 255, 0)') {
            console.warn('STILL NOT GREEN! Trying even more aggressive approach...');
          }
        }, 50);
      }
    });
  }
  
  // Run immediately
  setTimeout(fixMatchPercentageColor, 100);
  
  // Run again after a short delay
  setTimeout(fixMatchPercentageColor, 500);
  
  // Then keep checking every second - indefinitely
  setInterval(fixMatchPercentageColor, 1000);
  
  // Add MutationObserver to catch any dynamic changes
  const observer = new MutationObserver(function(mutations) {
    fixMatchPercentageColor();
  });
  
  // Start observing
  observer.observe(document.body, { 
    childList: true, 
    subtree: true,
    characterData: true
  });
  
  console.log("Enhanced match percentage fix script loaded and running!");
})();
*/
