import React, { useState, Suspense, useEffect } from 'react';
import Papa from 'papaparse';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, Switch, FormControlLabel, Typography, Tooltip, CircularProgress } from '@mui/material';

// Color palette for clusters
const CLUSTER_COLORS = [
  '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', 
  '#FF9F40', '#FF6384', '#C9CBCF', '#4CAF50'
];

function EmbeddingPoint({ point, showLabels, onPointClick, isSelected }) {
  // State for hover effect
  const [hovered, setHovered] = useState(false);
  
  // Handle pointer events
  const handlePointerOver = () => setHovered(true);
  const handlePointerOut = () => setHovered(false);
  const handleClick = () => onPointClick(point);
  
  // Optimize performance for large datasets
  const pointSize = isSelected ? 0.6 : (hovered ? 0.5 : 0.4);
  const pointOpacity = isSelected ? 1 : (hovered ? 0.95 : 0.85);
  const emissiveIntensity = isSelected ? 0.7 : (hovered ? 0.5 : 0.3);
  
  return (
    <group>
      <mesh 
        position={[point.x, point.y, point.z]}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
        onClick={handleClick}
        scale={isSelected ? 1.3 : (hovered ? 1.2 : 1)}
      >
        <sphereGeometry args={[pointSize]} />
        <meshStandardMaterial 
          color={CLUSTER_COLORS[point.cluster % CLUSTER_COLORS.length]} 
          opacity={pointOpacity} 
          transparent
          emissive={CLUSTER_COLORS[point.cluster % CLUSTER_COLORS.length]}
          emissiveIntensity={emissiveIntensity}
        />
      </mesh>
      {(showLabels || hovered || isSelected) && (
        <Text
          position={[point.x, point.y, point.z + 0.6]}
          fontSize={isSelected ? 0.4 : (hovered ? 0.35 : 0.3)}
          color="white"
          anchorX="center"
          anchorY="middle"
          backgroundColor={isSelected ? "#000000C0" : (hovered ? "#000000A0" : "#00000080")}
          padding={0.1}
        >
          {point.name}
        </Text>
      )}
    </group>
  );
}

function EmbeddingViewer() {
  const [darkMode, setDarkMode] = useState(true);
  const [embeddings, setEmbeddings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [showLabels, setShowLabels] = useState(false);

  useEffect(() => {
    const loadEmbeddings = async () => {
      try {
        // Attempt to load the tsne JSON data first
        try {
          console.log('Attempting to fetch embeddings from /data/game_summary_embeddings_tsne.json');
          const response = await fetch('/data/game_summary_embeddings_tsne.json');
          console.log('Fetch response status:', response.status, response.statusText);
          
          if (response.ok) {
            const jsonData = await response.json();
            console.log(`Loaded ${jsonData.length} embeddings from JSON`);
            console.log('First few embeddings:', jsonData.slice(0, 3));
            
            // Convert JSON data to our expected format
            const processedEmbeddings = jsonData.map((item, index) => ({
              id: index.toString(),
              name: item.title || `Game ${index}`,
              x: item.tsneX || 0,
              y: item.tsneY || 0,
              z: item.tsneZ || 0,
              cluster: item.cluster || 0,
              provider: item.provider || '',
              summary: item.summary || ''
            }));
            
            console.log('Processed embeddings:', processedEmbeddings.slice(0, 3));
            setEmbeddings(processedEmbeddings);
            setLoading(false);
            return;
          } else {
            console.error('Failed to fetch embeddings, response not OK');
          }
        } catch (jsonError) {
          console.warn('Could not load JSON embeddings, falling back to CSV', jsonError);
        }
        
        // Fallback to CSV format
        const response = await fetch('/game_embeddings.csv');
        const csvText = await response.text();
        
        // Custom parsing for the CSV
        const parseEmbeddings = (csvContent) => {
          // Basic CSV parsing - split by lines and then by commas
          const lines = csvContent.split('\n').filter(line => line.trim().length > 0);
          const embeddings = [];
          
          // Process each line
          lines.forEach((line, index) => {
            const values = line.split(',').map(val => isNaN(parseFloat(val)) ? val : parseFloat(val));
            
            // Assume first 3 values are tsne coordinates
            embeddings.push({
              id: index.toString(),
              name: `Game ${index}`,
              x: values[0] || 0, 
              y: values[1] || 0,
              z: values[2] || 0,
              cluster: index % 9, // Distribute across 9 clusters if not provided
              embedding: values
            });
          });

          console.log(`Total embeddings loaded: ${embeddings.length}`);
          setEmbeddings(embeddings);
          setLoading(false);
        };

        parseEmbeddings(csvText);
      } catch (error) {
        console.error('Error loading embeddings:', error);
        setLoading(false);
      }
    };

    loadEmbeddings();
  }, []);

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
    },
  });

  // Group embeddings by cluster
  const clusterGroups = embeddings.reduce((acc, point) => {
    if (!acc[point.cluster]) acc[point.cluster] = [];
    acc[point.cluster].push(point);
    return acc;
  }, {});

  // State for selected game details panel
  const [selectedGame, setSelectedGame] = useState(null);

  // Function to handle point click
  const handlePointClick = (point) => {
    setSelectedGame(point);
  };

  // Function to close the details panel
  const handleCloseDetails = () => {
    setSelectedGame(null);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, backgroundColor: darkMode ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={darkMode}
                  onChange={() => setDarkMode(!darkMode)}
                />
              }
              label={darkMode ? 'Dark Mode' : 'Light Mode'}
            />
            <FormControlLabel
              control={
                <Switch
                  checked={showLabels}
                  onChange={() => setShowLabels(!showLabels)}
                />
              }
              label="Show Labels"
            />
          </Box>
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
            <Typography variant="h6">
              Casino Game Embeddings: {embeddings.length} games
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
              {Object.keys(clusterGroups).map(clusterId => {
                const count = clusterGroups[clusterId].length;
                const percentage = Math.round((count / embeddings.length) * 100);
                return (
                  <Tooltip 
                    key={clusterId}
                    title={
                      <React.Fragment>
                        <Typography variant="subtitle1">Cluster {clusterId}</Typography>
                        <Typography variant="body2">{count} games ({percentage}%)</Typography>
                        {clusterGroups[clusterId].slice(0, 3).map(game => (
                          <Typography key={game.id} variant="body2" sx={{ opacity: 0.8 }}>
                            • {game.name || `Game ${game.id}`}
                          </Typography>
                        ))}
                        {count > 3 && <Typography variant="body2">...and {count - 3} more</Typography>}
                      </React.Fragment>
                    }
                    arrow
                  >
                    <Box
                      sx={{
                        width: 25,
                        height: 25,
                        backgroundColor: CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length],
                        cursor: 'pointer',
                        border: '2px solid',
                        borderColor: selectedCluster === parseInt(clusterId) ? '#fff' : 'transparent',
                        opacity: selectedCluster === null || selectedCluster === parseInt(clusterId) ? 1 : 0.3,
                        '&:hover': { opacity: 0.7, transform: 'scale(1.1)' },
                        transition: 'all 0.2s',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        borderRadius: '4px'
                      }}
                      onClick={() => setSelectedCluster(
                        selectedCluster === parseInt(clusterId) ? null : parseInt(clusterId)
                      )}
                    >
                      {selectedCluster === parseInt(clusterId) && (
                        <Typography sx={{ color: '#fff', fontWeight: 'bold', fontSize: '12px' }}>
                          ✓
                        </Typography>
                      )}
                    </Box>
                  </Tooltip>
                );
              })}
            </Box>
          </Box>
        </Box>
        
        {loading ? (
          <Box 
            sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              height: '100%' 
            }}
          >
            <CircularProgress />
          </Box>
        ) : (
          <Box sx={{ position: 'relative', flex: 1 }}>
            <Canvas 
              camera={{ position: [0, 0, 15] }}
              style={{ width: '100%', height: '100%' }}
              dpr={[1, 2]} // Optimize for performance
              performance={{ min: 0.5 }} // Adjust performance for large datasets
            >
              <color attach="background" args={[darkMode ? '#111111' : '#f5f5f5']} />
              <ambientLight intensity={0.7} />
              <pointLight position={[10, 10, 10]} intensity={1.5} />
              <OrbitControls 
                enablePan={true} 
                enableZoom={true} 
                enableRotate={true} 
                maxDistance={30}
                minDistance={2}
              />
              
              {/* Render points in batches for better performance */}
              <group>
                {embeddings
                  .filter(point => selectedCluster === null || point.cluster === selectedCluster)
                  .map((point, index) => {
                    // Scale coordinates to bring points closer together and add some spacing
                    const scaleFactor = 0.25; // Reduced scale factor for more space between points
                    const scaledPoint = {
                      ...point,
                      x: point.x * scaleFactor,
                      y: point.y * scaleFactor,
                      z: point.z * scaleFactor
                    };
                    
                    // Check if this point is the selected game
                    const isSelected = selectedGame && selectedGame.id === point.id;
                    
                    return (
                      <EmbeddingPoint 
                        key={point.id || index} 
                        point={scaledPoint} 
                        showLabels={showLabels}
                        onPointClick={handlePointClick}
                        isSelected={isSelected}
                      />
                    );
                  })}
              </group>
            </Canvas>
            
            {/* Cluster info panel */}
            {selectedCluster !== null && (
              <Box 
                sx={{ 
                  position: 'absolute', 
                  bottom: 20, 
                  left: 20, 
                  p: 2, 
                  backgroundColor: 'rgba(0,0,0,0.7)',
                  color: 'white',
                  borderRadius: 1,
                  maxWidth: '300px',
                  maxHeight: '300px',
                  overflowY: 'auto'
                }}
              >
                <Typography variant="h6">
                  Cluster {selectedCluster}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  {clusterGroups[selectedCluster]?.length || 0} games
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 'bold', mt: 1 }}>
                  Top Games in Cluster:
                </Typography>
                {clusterGroups[selectedCluster]?.slice(0, 5).map((game, idx) => (
                  <Typography key={idx} variant="body2" sx={{ mt: 0.5 }}>
                    • {game.name}
                  </Typography>
                ))}
              </Box>
            )}
            
            {/* Game details panel */}
            {selectedGame && (
              <Box 
                sx={{ 
                  position: 'absolute', 
                  top: 20, 
                  right: 20, 
                  p: 2, 
                  backgroundColor: 'rgba(0,0,0,0.85)',
                  color: 'white',
                  borderRadius: 1,
                  maxWidth: '400px',
                  maxHeight: '80%',
                  overflowY: 'auto',
                  boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
                  backdropFilter: 'blur(5px)'
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h6" sx={{ 
                    color: CLUSTER_COLORS[selectedGame.cluster % CLUSTER_COLORS.length],
                    textShadow: '0 0 5px rgba(0,0,0,0.5)'
                  }}>
                    {selectedGame.name}
                  </Typography>
                  <Box 
                    onClick={handleCloseDetails}
                    sx={{ cursor: 'pointer', p: 0.5, '&:hover': { opacity: 0.7 } }}
                  >
                    ✕
                  </Box>
                </Box>
                
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  mb: 2,
                  p: 1,
                  backgroundColor: `${CLUSTER_COLORS[selectedGame.cluster % CLUSTER_COLORS.length]}22`,
                  borderRadius: 1
                }}>
                  <Box 
                    sx={{ 
                      width: 20, 
                      height: 20, 
                      backgroundColor: CLUSTER_COLORS[selectedGame.cluster % CLUSTER_COLORS.length],
                      borderRadius: '50%',
                      mr: 1
                    }}
                  />
                  <Typography variant="body2">
                    <strong>Cluster {selectedGame.cluster}</strong> - {clusterGroups[selectedGame.cluster]?.length || 0} similar games
                  </Typography>
                </Box>
                
                {selectedGame.provider && (
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Provider:</strong> {selectedGame.provider}
                  </Typography>
                )}
                
                {selectedGame.summary && (
                  <>
                    <Typography variant="body2" sx={{ fontWeight: 'bold', mt: 2, borderBottom: '1px solid rgba(255,255,255,0.2)', pb: 0.5 }}>
                      Game Summary
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1, whiteSpace: 'pre-line', lineHeight: 1.5 }}>
                      {selectedGame.summary}
                    </Typography>
                  </>
                )}
                
                {/* Similar games section */}
                <Typography variant="body2" sx={{ fontWeight: 'bold', mt: 3, borderBottom: '1px solid rgba(255,255,255,0.2)', pb: 0.5 }}>
                  Similar Games
                </Typography>
                <Box sx={{ mt: 1 }}>
                  {clusterGroups[selectedGame.cluster]
                    ?.filter(game => game.id !== selectedGame.id)
                    .slice(0, 5)
                    .map((game, idx) => (
                      <Box 
                        key={idx} 
                        sx={{ 
                          p: 0.5, 
                          cursor: 'pointer', 
                          '&:hover': { backgroundColor: 'rgba(255,255,255,0.1)' },
                          borderRadius: 1,
                          mt: 0.5
                        }}
                        onClick={() => handlePointClick(game)}
                      >
                        <Typography variant="body2">
                          {game.name}
                          {game.provider && <span style={{ opacity: 0.7 }}> by {game.provider}</span>}
                        </Typography>
                      </Box>
                    ))
                  }
                </Box>
              </Box>
            )}
          </Box>
        )}
      </Box>
    </ThemeProvider>
  );
}

export default EmbeddingViewer;
