import React, { useState } from 'react';
import { DarkBlueText, MagentaText, GreenMatchText } from './TextColors';
// import CasinoHeading from './CasinoHeading';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import SearchIcon from '@mui/icons-material/Search';
import CircularProgress from '@mui/material/CircularProgress';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import Collapse from '@mui/material/Collapse';
// Import the new API service
import { performSearch } from './api-service';
import './App.css';

// Create a dark theme with purple accents as per user preferences
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#bb86fc',
    },
    secondary: {
      main: '#03dac6',
    },
    background: {
      default: '#121212',
      paper: '#1e1e2a', // Very dark card backgrounds as per user preference
    },
    text: {
      primary: '#e6e6f0',
      secondary: '#c0c7ff'
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 8px 16px rgba(0,0,0,0.5)', // Prominent shadows as per user preference
          borderRadius: '12px',
        }
      }
    },
    MuiTypography: {
      styleOverrides: {
        h5: {
          color: '#e6e6f0',
        },
        // No global body2 styles that might affect other elements
      }
    }
  }
});

// Sample suggestions for the search box
const SUGGESTIONS = [
  'Mythology',
  'Exciting',
  'Movie & TV',
  'Megaways',
  'Upbeat',
  'Sounds great',
  'Jackpot',
  'Adventure',
  'Magic'
];


  // Using styled components from TextColors.js instead of local definitions

  // StarsField component - adds gently twinkling stars to the background
  const StarsField = () => {
    const [stars, setStars] = useState([]);
    
    // Generate a set of random stars on component mount
    React.useEffect(() => {
      const numberOfStars = 100; // Adjust for more or fewer stars
      const newStars = [];
      
      for (let i = 0; i < numberOfStars; i++) {
        // Generate random properties for each star
        newStars.push({
          id: i,
          left: `${Math.random() * 100}%`,
          top: `${Math.random() * 100}%`,
          size: 1 + Math.random() * 2, // Size between 1-3px
          opacity: 0.2 + Math.random() * 0.5, // Opacity between 0.2-0.7
          duration: 3 + Math.random() * 7, // Animation duration between 3-10s
          delay: Math.random() * 10 // Random delay for each star
        });
      }
      
      setStars(newStars);
    }, []);
    
    return (
      <div className="stars-container">
        {stars.map(star => (
          <div
            key={star.id}
            className="star"
            style={{
              left: star.left,
              top: star.top,
              width: `${star.size}px`,
              height: `${star.size}px`,
              '--twinkle-opacity': star.opacity,
              '--twinkle-duration': `${star.duration}s`,
              '--twinkle-delay': `${star.delay}s`
            }}
          />
        ))}
      </div>
    );
  };

  // ParallaxCard component - adds a cool 3D tilt effect to cards based on mouse position
  const ParallaxCard = ({ children }) => {
    const [transform, setTransform] = useState({
      rotateX: 0,
      rotateY: 0,
      translateZ: 0,
      scale: 1,
      brightness: 1
    });
    
    const handleMouseMove = (e) => {
      const card = e.currentTarget;
      const rect = card.getBoundingClientRect();
      
      // Calculate mouse position relative to card center
      const x = e.clientX - rect.left - rect.width / 2;
      const y = e.clientY - rect.top - rect.height / 2;
      
      // Calculate rotation based on mouse position (max rotation: 8deg)
      const rotateY = (x / rect.width) * 8;
      const rotateX = -(y / rect.height) * 8;
      
      // Update transform state
      setTransform({
        rotateX,
        rotateY,
        translateZ: 10,
        scale: 1.03,
        brightness: 1.03
      });
    };
    
    const handleMouseLeave = () => {
      // Reset transform on mouse leave
      setTransform({
        rotateX: 0,
        rotateY: 0,
        translateZ: 0,
        scale: 1,
        brightness: 1
      });
    };
    
    return (
      <Box
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        sx={{
          transformStyle: 'preserve-3d',
          transition: 'transform 0.1s ease, filter 0.1s ease',
          transform: `perspective(1000px) rotateX(${transform.rotateX}deg) rotateY(${transform.rotateY}deg) translateZ(${transform.translateZ}px) scale(${transform.scale})`,
          filter: `brightness(${transform.brightness})`,
          '&:hover': {
            boxShadow: '0 10px 20px rgba(0,0,0,0.6)'
          }
        }}
      >
        {children}
      </Box>
    );
  };

  // ExpandableText component that shows the first 2 lines of text with a dropdown to see the full text
  const ExpandableText = ({ text }) => {
    const [expanded, setExpanded] = useState(false);
    
    // Calculate if text is likely to be more than 2 lines (rough estimate: ~100 chars per line)
    const isLongText = text && text.length > 200;
    
    // Get the truncated preview text (first 2 lines or ~200 chars)
    const getPreviewText = () => {
      if (!text) return '';
      
      // Try to find line breaks and split by those first
      const lines = text.split('\n');
      if (lines.length > 1) {
        return lines.slice(0, 2).join('\n') + (lines.length > 2 ? '...' : '');
      }
      
      // Otherwise estimate by characters
      return text.substring(0, 200) + (text.length > 200 ? '...' : '');
    };
    
    return (
      <Box sx={{ mt: 0.5 }}>
        {/* Always visible preview text */}
        {!expanded && (
          <Typography variant="body2" sx={{ mt: 0.5 }}>
            {getPreviewText()}
          </Typography>
        )}
        
        {/* Expandable full text */}
        <Collapse in={expanded} timeout="auto">
          <Typography variant="body2" sx={{ mt: 0.5 }}>
            {text}
          </Typography>
        </Collapse>
        
        {/* Only show expand/collapse button if text is long enough */}
        {isLongText && (
          <Box 
            sx={{ 
              display: 'flex', 
              justifyContent: 'center',
              mt: 0.5,
              cursor: 'pointer',
              color: '#bb86fc',
              '&:hover': { opacity: 0.8 }
            }}
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? (
              <Box sx={{ display: 'flex', alignItems: 'center', fontSize: '0.8rem' }}>
                <ExpandLessIcon fontSize="small" sx={{ mr: 0.5 }} />
                <span>Show less</span>
              </Box>
            ) : (
              <Box sx={{ display: 'flex', alignItems: 'center', fontSize: '0.8rem' }}>
                <ExpandMoreIcon fontSize="small" sx={{ mr: 0.5 }} />
                <span>Read more</span>
              </Box>
            )}
          </Box>
        )}
      </Box>
    );
  };

  // Process search results to ensure correct titles and generate brief summaries
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
      
      // Generate a brief summary if it doesn't exist
      if (!result.short_summary && result.summary) {
        // Get first sentence or first 100 characters, whichever is shorter
        const firstSentenceMatch = result.summary.match(/^(.*?[.!?])(\s|$)/);  
        result.short_summary = firstSentenceMatch 
          ? firstSentenceMatch[1]
          : result.summary.slice(0, 100) + (result.summary.length > 100 ? '...' : '');
      }
      
      return result;
    });
  };

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState(null);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [semanticOnly, setSemanticOnly] = useState(false);
  const [showShortSummary, setShowShortSummary] = useState(false);
  
  // Add state mounting indicator to debug render issues
  const [isMounted, setIsMounted] = useState(false);
  
  React.useEffect(() => {
    setIsMounted(true);
    console.log('App component mounted');
    return () => {
      console.log('App component unmounted');
      setIsMounted(false);
    };
  }, []);

  // Real search: call the Flask backend
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && searchQuery.trim()) {
      handleSearch(searchQuery);
    }
  };

  const handleSearch = async (query) => {
    setIsSearching(true);
    setError(null);
    setShowSuggestions(false);
    
    console.log('Searching for:', query, 'Semantic only:', semanticOnly);
    
    try {
      // Use our dedicated API service
      const data = await performSearch(query, semanticOnly);
      
      // Extract results
      const results = data && data.results ? data.results : [];
      console.log(`Received ${results.length} search results`);
      
      
      // Set the search results
      setSearchResults(results);
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
      
      // More detailed error handling
      let errorMessage = 'Failed to search. Please try again.';
      
      if (error.response) {
        // Server responded with error
        console.error('API error response:', error.response);
        errorMessage = `API Error (${error.response.status}): ${error.response.statusText || 'Unknown error'}`;
        if (error.response.data && error.response.data.error) {
          errorMessage = `API Error: ${error.response.data.error}`;
        }
      } else if (error.request) {
        // Request was made but no response received (network error)
        console.error('No response received:', error.request);
        errorMessage = 'Network error: Unable to reach search API. Please check your connection.';
      } else {
        // Something else caused the error
        console.error('Error details:', error.message);
        errorMessage = `Error: ${error.message}`;
      }
      
      setError(errorMessage);
    } finally {
      setIsSearching(false);
      console.log('Search completed');
    }
  };

  
  const handleSuggestionClick = (suggestion) => {
    // Clear previous results first to ensure fresh data
    setSearchResults([]);
    // Set a timeout to ensure state updates before search
    setTimeout(() => {
      setSearchQuery(suggestion);
      handleSearch(suggestion);
    }, 50);
  };
;

  const handleInputChange = (e) => {
    setSearchQuery(e.target.value);
    setShowSuggestions(true);
    setSearchResults([]);
    setError(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && searchQuery.trim()) {
      handleSearch(searchQuery);
    }
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setSearchResults([]);
    setShowSuggestions(true);
  };

  console.log('Rendering App component, mounted:', isMounted);
  
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box
        className="main-container"
        sx={{
          backgroundColor: 'black',
          minHeight: '100vh',
          color: '#f0f4ff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 4,
          background: 'radial-gradient(circle at center, rgba(40, 5, 60, 0.4) 0%, rgba(10, 2, 20, 0.95) 70%, black 100%)',
          position: 'relative',
          overflow: 'hidden'
        }}>
        <Box 
          className="search-container"
          sx={{
            width: 370,
            borderRadius: 4,
            background: 'linear-gradient(135deg, rgba(90, 20, 90, 0.95) 0%, rgba(40, 5, 60, 0.98) 50%, rgba(15, 2, 30, 0.99) 100%)',
            boxShadow: '0 15px 50px rgba(0,0,0,0.75)',
            p: 3,
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'stretch',
            overflow: 'hidden'
          }}>
          {/* Add the sparkling stars in the search container's purple gradient background */}
          <StarsField />
          <Box sx={{ mb: 3, display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', position: 'relative', zIndex: 1 }}>
            <Box sx={{ mb: 1 }}>
              <Typography variant="overline" sx={{ 
                color: '#f0f4ff', 
                letterSpacing: 1,
                display: 'inline-block',
                border: '1px solid #ff4081',
                borderRadius: '4px',
                padding: '0px 4px',
                fontSize: '0.6rem',
                fontWeight: 'bold',
                lineHeight: 1.5
              }}>
                BETA
              </Typography>
            </Box>
            <Typography variant="h5" sx={{ fontWeight: 700, mb: 1 }} style={{color: '#ff4081'}}>
              Casino Search
            </Typography>
            <DarkBlueText variant="body2">
              Ask for vibes, themes, titles — whatever you're craving!
            </DarkBlueText>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, mb: 2 }}>
              {/* Only show Return to Search after search results are displayed */}
              {searchResults.length > 0 && (
                <Typography 
                  variant="body2"
                  onClick={() => { setSearchQuery(''); setShowSuggestions(true); setSearchResults([]); }}
                  style={{ color: '#f0f4ff', cursor: 'pointer' }}
                >
                  Return to Search
                </Typography>
              )}
              
              {/* Toggle controls for Semantic and Summary Type */}
              <Box sx={{ display: 'flex', mt: 1, mb: 1.5, justifyContent: 'center', gap: 2 }}>
                {/* Toggle Switch for Semantic-Only Mode */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box 
                    onClick={() => setSemanticOnly(!semanticOnly)}
                    sx={{
                      width: '40px',
                      height: '20px',
                      bgcolor: semanticOnly ? '#ff4081' : 'rgba(255,255,255,0.3)',
                      borderRadius: '10px',
                      position: 'relative',
                      cursor: 'pointer',
                      transition: 'background-color 0.3s',
                      display: 'flex',
                      alignItems: 'center',
                      padding: '0 2px'
                    }}
                  >
                    <Box 
                      sx={{
                        width: '16px',
                        height: '16px',
                        borderRadius: '50%',
                        bgcolor: '#f0f4ff',
                        position: 'absolute',
                        left: semanticOnly ? '22px' : '2px',
                        transition: 'left 0.3s'
                      }}
                    />
                  </Box>
                  <Typography variant="caption" sx={{ color: semanticOnly ? '#f0f4ff' : 'rgba(240,244,255,0.5)' }}>
                    Semantic
                  </Typography>
                </Box>
                
                {/* Toggle Switch for Brief/Detailed Summary */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box 
                    onClick={() => setShowShortSummary(!showShortSummary)}
                    sx={{
                      width: '40px',
                      height: '20px',
                      bgcolor: showShortSummary ? '#ff4081' : 'rgba(255,255,255,0.3)',
                      borderRadius: '10px',
                      position: 'relative',
                      cursor: 'pointer',
                      transition: 'background-color 0.3s',
                      display: 'flex',
                      alignItems: 'center',
                      padding: '0 2px'
                    }}
                  >
                    <Box 
                      sx={{
                        width: '16px',
                        height: '16px',
                        borderRadius: '50%',
                        bgcolor: '#f0f4ff',
                        position: 'absolute',
                        left: showShortSummary ? '22px' : '2px',
                        transition: 'left 0.3s'
                      }}
                    />
                  </Box>
                  <Typography variant="caption" sx={{ color: showShortSummary ? '#f0f4ff' : 'rgba(240,244,255,0.5)' }}>
                    {showShortSummary ? 'Brief' : 'Detailed'}
                  </Typography>
                </Box>
              </Box>
            </Box>

          </Box>

          {/* Suggestions */}
          {showSuggestions && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mb: 2 }}>
              {/* First Row */}
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                {SUGGESTIONS.slice(0, 3).map((suggestion, idx) => (
                  <Chip
                    key={idx}
                    label={suggestion}
                    onClick={() => handleSuggestionClick(suggestion)}
                    sx={{
                      borderRadius: '30px',
                      fontWeight: 500,
                      fontSize: '0.8rem',
                      px: 1.5,
                      py: 0.4,
                      height: 'auto',
                      bgcolor: 'rgba(255, 255, 255, 0.15)',
                      color: '#f0f4ff',
                      border: 'none',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                      transition: 'all 0.2s ease',
                      '&:hover': { 
                        bgcolor: 'rgba(255, 64, 129, 0.8)', 
                        color: '#f0f4ff',
                      },
                    }}
                  />
                ))}
              </Box>
              
              {/* Second Row */}
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                {SUGGESTIONS.slice(3, 6).map((suggestion, idx) => (
                  <Chip
                    key={idx + 3}
                    label={suggestion}
                    onClick={() => handleSuggestionClick(suggestion)}
                    sx={{
                      borderRadius: '30px',
                      fontWeight: 500,
                      fontSize: '0.8rem',
                      px: 1.5,
                      py: 0.4,
                      height: 'auto',
                      bgcolor: 'rgba(255, 255, 255, 0.15)',
                      color: '#f0f4ff',
                      border: 'none',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                      transition: 'all 0.2s ease',
                      '&:hover': { 
                        bgcolor: 'rgba(255, 64, 129, 0.8)', 
                        color: '#f0f4ff',
                      },
                    }}
                  />
                ))}
              </Box>
              
              {/* Third Row */}
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                {SUGGESTIONS.slice(6).map((suggestion, idx) => (
                  <Chip
                    key={idx + 6}
                    label={suggestion}
                    onClick={() => handleSuggestionClick(suggestion)}
                    sx={{
                      borderRadius: '30px',
                      fontWeight: 500,
                      fontSize: '0.8rem',
                      px: 1.5,
                      py: 0.4,
                      height: 'auto',
                      bgcolor: 'rgba(255, 255, 255, 0.15)',
                      color: '#f0f4ff',
                      border: 'none',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                      transition: 'all 0.2s ease',
                      '&:hover': { 
                        bgcolor: 'rgba(255, 64, 129, 0.8)', 
                        color: '#f0f4ff',
                      },
                    }}
                  />
                ))}
              </Box>
            </Box>
          )}

          {/* Search Box with Fiery Border */}
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            bgcolor: 'rgba(255,255,255,0.06)',
            borderRadius: 3,
            px: 2,
            py: 1.5,
            mb: 2,
            position: 'relative',
            width: '100%',
            '&::before': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              borderRadius: 3,
              padding: '2px',
              background: 'linear-gradient(45deg, #ff4500, #ff8d00, #ffcc00, #ff4500)',
              WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
              WebkitMaskComposite: 'xor',
              maskComposite: 'exclude',
              boxSizing: 'border-box',
              opacity: searchQuery ? 1 : 0.6,
              transition: 'opacity 0.3s',
              animation: searchQuery ? 'fire-border 2s infinite' : 'none',
            },
            '@keyframes fire-border': {
              '0%': { backgroundPosition: '0% 50%' },
              '50%': { backgroundPosition: '100% 50%' },
              '100%': { backgroundPosition: '0% 50%' },
            },
          }}>
            <TextField
              fullWidth
              variant="standard"
              placeholder="What are you in the mood to play?"
              value={searchQuery}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              InputProps={{
                disableUnderline: true,
                style: { color: '#f0f4ff', fontSize: '0.9rem', fontWeight: 500 },
              }}
              sx={{ 
                bgcolor: 'transparent',
                '& .MuiInputBase-input::placeholder': {
                  color: 'rgba(240, 244, 255, 0.8)',
                  opacity: 1,
                  fontSize: '0.85rem'
                },
                '& .MuiInputBase-root': {
                  width: '100%'
                }
              }}
            />
            <IconButton
              color="primary"
              onClick={() => handleSearch(searchQuery)}
              disabled={!searchQuery.trim() || isSearching}
              sx={{ ml: 1 }}
            >
              <SearchIcon />
            </IconButton>
          </Box>

          {/* Enhanced Loading indicator */}
          {isSearching && (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column', mt: 2 }}>
              <Box sx={{
                position: 'relative',
                display: 'inline-flex',
                '& .MuiCircularProgress-circle': {
                  strokeLinecap: 'round',
                }
              }}>
                <CircularProgress 
                  variant="determinate" 
                  value={100} 
                  size={40} 
                  thickness={4}
                  sx={{
                    color: 'rgba(40, 40, 55, 0.3)',
                  }}
                />
                <CircularProgress
                  size={40}
                  thickness={4}
                  sx={{
                    color: 'rgba(140, 80, 200, 0.7)',
                    position: 'absolute',
                    left: 0,
                    '& .MuiCircularProgress-circle': {
                      strokeDasharray: '150, 200',
                      strokeDashoffset: 0,
                    }
                  }}
                />
              </Box>
              <Typography variant="body2" sx={{ mt: 1, color: '#f0f4ff', opacity: 0.8, fontSize: '0.75rem' }}>
                Finding games...
              </Typography>
            </Box>
          )}

          {/* Add custom scrollbar styling */}
          <style>
            {
              `
              /* Text color override */
              body * {
                color: #f0f4ff !important;
              }

              /* Casino Search title specific override */
              h5.MuiTypography-root {
                color: #f0f4ff !important;
              }

              /* Subtitle specific override */
              p.MuiTypography-body2 {
                color: #f0f4ff !important;
              }
              
              /* Custom scrollbar styling */
              .results-container::-webkit-scrollbar {
                width: 8px;
              }
              .results-container::-webkit-scrollbar-track {
                background: rgba(20, 20, 25, 0.6);
                border-radius: 5px;
              }
              .results-container::-webkit-scrollbar-thumb {
                background: linear-gradient(to bottom, rgba(80, 80, 95, 0.8), rgba(50, 50, 65, 0.8));
                border-radius: 5px;
                box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.3);
              }
              .results-container::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(to bottom, rgba(100, 100, 115, 0.9), rgba(70, 70, 85, 0.9));
              }
              
              /* Fade-in animation */
              @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
              }
              .game-card {
                animation: fadeIn 0.3s ease-out;
                animation-fill-mode: both;
              }
              .game-card:nth-child(1) { animation-delay: 0.05s; }
              .game-card:nth-child(2) { animation-delay: 0.1s; }
              .game-card:nth-child(3) { animation-delay: 0.15s; }
              .game-card:nth-child(4) { animation-delay: 0.2s; }
              .game-card:nth-child(5) { animation-delay: 0.25s; }
              .game-card:nth-child(6) { animation-delay: 0.3s; }
              .game-card:nth-child(7) { animation-delay: 0.35s; }
              .game-card:nth-child(8) { animation-delay: 0.4s; }
              .game-card:nth-child(9) { animation-delay: 0.45s; }
              .game-card:nth-child(10) { animation-delay: 0.5s; }
              `
            }
          </style>
          
          {/* Display search results */}
          {!isSearching && searchResults.length > 0 && (
            <Box 
              className="results-container"
              sx={{ 
                mt: 2, 
                maxHeight: '400px', 
                overflowY: 'auto',
                borderRadius: 3,
                padding: '2px',
                background: 'linear-gradient(45deg, rgba(40,40,50,0.3), rgba(60,60,70,0.3))',
                '&:hover': {
                  background: 'linear-gradient(45deg, rgba(50,50,60,0.4), rgba(70,70,80,0.4))',
                }
              }}
            >
              {searchResults.map((game, index) => (
                <ParallaxCard key={index}>
                  <Card 
                    className="game-card"
                    sx={{ 
                      bgcolor: '#171720', 
                      color: '#f0f4ff', 
                      mb: 2, 
                      mt: index === 0 ? 0.5 : 0,
                      mx: 0.5,
                      borderRadius: 3,
                      border: '1px solid rgba(60, 60, 70, 0.4)',
                      boxShadow: '0 8px 30px rgba(15, 15, 20, 0.6)',
                      overflow: 'hidden'
                    }}
                  >
                    <CardContent>
                      {/* Game title */}
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {game.game_name || game.title || 'Unknown Game'}
                      </Typography>
                    
                    {/* Match percentage with dedicated green component */}
                    <GreenMatchText>
                      {Math.round(game.similarity * 100)}% match
                    </GreenMatchText>
                    
                    {/* Developer and Volatility section with decorative lines */}
                    <Box sx={{ mt: 0.5, mb: 1 }}>
                      {/* Top line */}
                      <Box sx={{ 
                        height: '1px', 
                        background: 'linear-gradient(to right, rgba(40, 40, 50, 0), rgba(80, 80, 100, 0.4), rgba(40, 40, 50, 0))',
                        mb: 1.5 
                      }} />
                      
                      {/* Developer and Volatility chips */}
                      <Box sx={{ 
                        display: 'flex',
                        justifyContent: 'center', 
                      }}>
                        <Chip 
                          label={game.provider || 'Unknown'}
                          size="small" 
                          sx={{ 
                            mr: 1, 
                            bgcolor: '#ffffff !important',
                            color: '#000000 !important',
                            fontWeight: 600,
                            border: '1px solid #444444',
                            '& .MuiChip-label': { 
                              fontSize: '0.7rem',
                              color: '#000000 !important',
                              fontWeight: '600 !important'
                            }
                          }}
                          style={{
                            backgroundColor: '#ffffff',
                            color: '#000000'
                          }}
                          className="developer-chip"
                        />
                        <Chip 
                          label={`Volatility: ${game.volatility || 'Unknown'}`}
                          size="small" 
                          sx={{ 
                            bgcolor: 
                              game.volatility?.toLowerCase().includes('high') && !game.volatility?.toLowerCase().includes('medium') ? 'rgba(244, 67, 54, 0.8)' : // High = red
                              game.volatility?.toLowerCase().includes('medium') && game.volatility?.toLowerCase().includes('high') ? 'rgba(255, 152, 0, 0.8)' : // Medium/High = orange
                              game.volatility?.toLowerCase().includes('medium') ? 'rgba(255, 140, 0, 0.8)' : // Medium = orange (was yellow)
                              'rgba(40, 45, 55, 0.8)', // Default dark background
                            color: 
                              game.volatility?.toLowerCase().includes('medium') && !game.volatility?.toLowerCase().includes('high') ? '#ffffff' : // White text on orange background (was black on yellow)
                              '#ffffff', // White text on other backgrounds
                            fontWeight: 700, // Bolder for better readability
                            '& .MuiChip-label': { fontSize: '0.7rem' }
                          }} 
                        />
                      </Box>
                      
                      {/* Bottom line */}
                      <Box sx={{ 
                        height: '1px', 
                        background: 'linear-gradient(to right, rgba(40, 40, 50, 0), rgba(80, 80, 100, 0.4), rgba(40, 40, 50, 0))',
                        mt: 1.5 
                      }} />
                    </Box>
                    
                    {/* Game summary with expandable text */}
                    <ExpandableText 
                      text={showShortSummary 
                        ? (game.short_summary || game.summary.split('.')[0] + '.') 
                        : game.summary} 
                    />
                    </CardContent>
                  </Card>
                </ParallaxCard>
              ))}
            </Box>
          )}
          
          {/* Show error message if any */}
          {error && (
            <Typography color="error" sx={{ mt: 2, textAlign: 'center' }}>
              {error}
            </Typography>
          )}
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
