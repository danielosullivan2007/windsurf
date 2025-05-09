import React, { useState, useEffect } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CircularProgress from '@mui/material/CircularProgress';
import axios from 'axios';
import './App.css';

// Create dark theme with purple accents as per user preferences
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
    }
  }
});

function AppSimple() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState(null);

  // Handle search input change
  const handleInputChange = (e) => {
    setSearchQuery(e.target.value);
  };

  // Handle search form submission
  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    setError(null);
    
    try {
      console.log('Searching for:', searchQuery);
      
      // Send request to API
      const response = await axios.post('http://127.0.0.1:5001/api/search', { 
        query: searchQuery
      });
      
      console.log('API response:', response.data);
      
      // Update state with results
      setSearchResults(response.data.results || []);
      
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
      setError('Failed to search. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box
        sx={{
          backgroundColor: 'black',
          minHeight: '100vh',
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 4,
          background: 'radial-gradient(circle at center, rgba(40, 5, 60, 0.4) 0%, rgba(10, 2, 20, 0.95) 70%, black 100%)'
        }}
      >
        <Box 
          sx={{
            width: 500,
            borderRadius: 4,
            background: 'linear-gradient(135deg, rgba(90, 20, 90, 0.95) 0%, rgba(40, 5, 60, 0.98) 50%, rgba(15, 2, 30, 0.99) 100%)',
            boxShadow: '0 15px 50px rgba(0,0,0,0.75)',
            p: 3,
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'stretch',
          }}
        >
          <Typography variant="h5" sx={{ fontWeight: 700, mb: 2, color: 'white', textAlign: 'center' }}>
            Casino Game Search
          </Typography>
          
          {/* Search form */}
          <Box component="form" onSubmit={handleSearch} sx={{ mb: 3 }}>
            <TextField
              fullWidth
              label="What are you in the mood to play?"
              variant="outlined"
              value={searchQuery}
              onChange={handleInputChange}
              sx={{ mb: 2 }}
            />
            <Button 
              type="submit" 
              variant="contained" 
              color="primary" 
              fullWidth
              disabled={isSearching}
            >
              {isSearching ? 'Searching...' : 'Search'}
            </Button>
          </Box>
          
          {/* Loading indicator */}
          {isSearching && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          )}
          
          {/* Error message */}
          {error && (
            <Typography color="error" sx={{ mb: 2, textAlign: 'center' }}>
              {error}
            </Typography>
          )}
          
          {/* Search results */}
          {!isSearching && searchResults.length > 0 ? (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Results ({searchResults.length})
              </Typography>
              
              <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                {searchResults.map((game, index) => (
                  <Card key={index} sx={{ mb: 2, bgcolor: '#1e1e2a' }}>
                    <CardContent>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {game.title || game.game_name || 'Unknown Game'}
                      </Typography>
                      
                      {/* Show developer name without prefix as per user preference */}
                      <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', mt: 1 }}>
                        {game.provider || 'Unknown'}
                      </Typography>
                      
                      {/* Show volatility with prefix as per user preference */}
                      <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', mt: 0.5 }}>
                        Volatility: {game.volatility || 'Unknown'}
                      </Typography>
                      
                      {game.summary && (
                        <Typography variant="body2" sx={{ mt: 1.5, color: 'rgba(255,255,255,0.9)' }}>
                          {game.summary}
                        </Typography>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </Box>
            </Box>
          ) : (!isSearching && searchQuery && (
            <Typography sx={{ textAlign: 'center', mt: 2 }}>
              No results found. Try a different search.
            </Typography>
          ))}
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default AppSimple;
