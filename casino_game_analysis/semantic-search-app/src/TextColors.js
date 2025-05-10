// Custom components with guaranteed styling using direct inline styles
import React from 'react';
import Typography from '@mui/material/Typography';

// Text for "Ask for vibes" paragraph - now ice blue
export const DarkBlueText = (props) => (
  <Typography
    variant="body2"
    {...props}
    style={{
      color: '#a0c4ff',
      marginBottom: '16px',
      maxWidth: '90%',
      ...props.style
    }}
  />
);

// Magenta text for "Switch to Standard Search" using direct inline approach
export const MagentaText = (props) => (
  <Typography
    variant="body2"
    {...props}
    style={{
      color: '#ff4081',
      cursor: 'pointer',
      fontWeight: 500,
      ...props.style
    }}
  />
);

// Vibrant green text for match percentages with enhanced styling to guarantee it works
export const GreenMatchText = (props) => (
  <Typography
    variant="body2"
    data-component-name="GreenMatchText-root"
    {...props}
    style={{
      color: '#00ff00 !important', // Vibrant green with !important flag
      WebkitTextFillColor: '#00ff00 !important', // For webkit browsers
      fontWeight: 600,
      fontSize: '0.8rem',
      textAlign: 'center',
      marginBottom: '4px',
      ...props.style
    }}
    className={`green-match-text ${props.className || ''}`}
  />
);
