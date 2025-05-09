import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Grid } from '@react-three/drei';
import { Box, FormControl, InputLabel, MenuItem, Select, FormGroup, FormControlLabel, Switch, Typography } from '@mui/material';
import { GameEmbedding } from '../types/GameEmbedding';
import * as THREE from 'three';
import { PointGlow } from './PointGlow';

// Color palette for clusters with bright, highly saturated colors
const CLUSTER_COLORS = [
  '#FF3B30', // Bright Red
  '#34C759', // Bright Green
  '#007AFF', // Bright Blue
  '#FFCC00', // Bright Yellow
  '#FF9500', // Bright Orange
  '#5856D6', // Bright Purple
  '#AF52DE', // Bright Magenta
  '#00C7BE', // Bright Teal
  '#FF2D55', // Bright Pink
];

interface PointProps {
  position: [number, number, number];
  color: string;
  gameTitle: string;
  cluster: number;
  showLabels: boolean;
}

function Point({ position, color, gameTitle, cluster, showLabels }: PointProps) {
  const [hovered, setHovered] = useState(false);
  const meshRef = useRef<THREE.Mesh>(null);
  
  return (
    <group>
      {/* Outer glow effect */}
      <PointGlow position={position} color={color} scale={1.8} />
      
      {/* Main point */}
      <mesh
        ref={meshRef}
        position={position}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[0.4, 32, 32]} />
        <meshBasicMaterial 
          color={color}
          toneMapped={false}
        />
      </mesh>
      
      {/* Label */}
      {(hovered || showLabels) && (
        <group position={[position[0], position[1] + 0.5, position[2]]}>
          <Text
            fontSize={0.15}
            color="white"
            anchorX="center"
            anchorY="middle"
            outlineWidth={0.01}
            outlineColor="black"
          >
            {gameTitle} (Cluster {cluster})
          </Text>
        </group>
      )}
    </group>
  );
}

interface GameEmbeddingsProps {
  embeddings: GameEmbedding[];
}

const GameEmbeddingsScene: React.FC<GameEmbeddingsProps> = ({ embeddings }) => {
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [showLabels, setShowLabels] = useState(false);
  
  const filteredEmbeddings = selectedCluster !== null 
    ? embeddings.filter(emb => emb.cluster === selectedCluster)
    : embeddings;
    
  return (
    <>
      <ambientLight intensity={1} />
      <directionalLight position={[10, 10, 10]} intensity={1.5} />
      <pointLight position={[0, 0, 5]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={1} color="#ffffff" />
      <hemisphereLight intensity={0.8} color="#ffffff" groundColor="#323232" />
      {/* No fog to ensure colors are visible at distance */}
      
      <Grid 
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#6f6f6f"
        position={[0, -0.01, 0]}
        args={[100, 100]}
        sectionSize={3}
        sectionThickness={1}
        sectionColor="#9d4b4b"
        fadeDistance={50}
        fadeStrength={1}
        infiniteGrid
      />
      
      {filteredEmbeddings.map((game, index) => (
        <Point 
          key={game.id || index}
          position={[game.tsneX * 3, game.tsneY * 3, game.tsneZ * 3]}
          color={CLUSTER_COLORS[game.cluster % CLUSTER_COLORS.length]}
          gameTitle={game.title}
          cluster={game.cluster}
          showLabels={showLabels}
        />
      ))}
      
      <OrbitControls />
    </>
  );
};

const GameEmbeddingsVisualization: React.FC = () => {
  const [embeddings, setEmbeddings] = useState<GameEmbedding[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [showLabels, setShowLabels] = useState(false);
  const [uniqueClusters, setUniqueClusters] = useState<number[]>([]);
  
  useEffect(() => {
    const fetchEmbeddings = async () => {
      try {
        setLoading(true);
        const response = await fetch('/game_embeddings_tsne.json');
        const data = await response.json();
        
        // Add an id field if it doesn't exist
        const processedData = data.map((item: any, index: number) => ({
          ...item,
          id: item.id || `game-${index}`
        }));
        
        setEmbeddings(processedData);
        
        // Extract unique clusters
        const clusters = Array.from(new Set(processedData.map((item: GameEmbedding) => item.cluster))) as number[];
        setUniqueClusters(clusters);
        
        setLoading(false);
      } catch (error) {
        console.error('Error fetching embeddings:', error);
        setLoading(false);
      }
    };
    
    fetchEmbeddings();
  }, []);
  
  const handleClusterChange = (event: any) => {
    const value = event.target.value;
    setSelectedCluster(value === 'all' ? null : Number(value));
  };
  
  const handleLabelToggle = () => {
    setShowLabels(!showLabels);
  };
  
  const filteredEmbeddings = selectedCluster !== null 
    ? embeddings.filter(emb => emb.cluster === selectedCluster)
    : embeddings;
  
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', bgcolor: '#121212', color: 'white' }}>
      <Box sx={{ p: 2, borderBottom: '1px solid #333' }}>
        <Typography variant="h4" gutterBottom>
          Casino Game Embeddings Visualization
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel id="cluster-select-label">Filter by Cluster</InputLabel>
            <Select
              labelId="cluster-select-label"
              value={selectedCluster === null ? 'all' : selectedCluster}
              label="Filter by Cluster"
              onChange={handleClusterChange}
              sx={{ bgcolor: '#333', color: 'white' }}
            >
              <MenuItem value="all">All Clusters</MenuItem>
              {uniqueClusters.map(cluster => (
                <MenuItem key={cluster} value={cluster}>
                  <Box 
                    component="span" 
                    sx={{ 
                      display: 'inline-block', 
                      width: 16, 
                      height: 16, 
                      borderRadius: '50%', 
                      bgcolor: CLUSTER_COLORS[cluster % CLUSTER_COLORS.length],
                      mr: 1
                    }} 
                  />
                  Cluster {cluster}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <FormGroup>
            <FormControlLabel
              control={<Switch checked={showLabels} onChange={handleLabelToggle} />}
              label="Show All Labels"
            />
          </FormGroup>
        </Box>
      </Box>
      
      <Box sx={{ flex: 1, position: 'relative' }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography>Loading embeddings...</Typography>
          </Box>
        ) : (
          <Canvas camera={{ position: [2, 2, 5], fov: 50 }}>
            <GameEmbeddingsScene 
              embeddings={filteredEmbeddings} 
            />
          </Canvas>
        )}
      </Box>
    </Box>
  );
};

export default GameEmbeddingsVisualization;
