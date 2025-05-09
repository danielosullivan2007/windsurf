import React, { useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Component for visualizing game embeddings in 3D space
function GameEmbeddingsVisualization({ embeddings, selectedCluster }) {
  // Create references for points and geometries
  const pointsRef = React.useRef();
  
  // Use memo to avoid recalculating positions on every render
  const { positions, colors } = useMemo(() => {
    // Initialize positions and colors arrays
    const positions = new Float32Array(embeddings.length * 3);
    const colors = new Float32Array(embeddings.length * 3);
    
    // Set color palette for clusters
    const clusterColors = [
      [1.0, 0.4, 0.5], // pink
      [0.2, 0.6, 0.9], // blue
      [1.0, 0.8, 0.3], // yellow
      [0.3, 0.75, 0.75], // teal
      [0.6, 0.4, 1.0], // purple
      [1.0, 0.6, 0.25], // orange
      [0.3, 0.8, 0.3], // green
      [0.8, 0.8, 0.8], // gray
      [0.9, 0.2, 0.2]  // red
    ];
    
    // Fill positions and colors arrays
    embeddings.forEach((point, i) => {
      positions[i * 3] = point.x; // x
      positions[i * 3 + 1] = point.y; // y
      positions[i * 3 + 2] = point.z; // z
      
      // Assign color based on cluster
      const colorIndex = point.cluster % clusterColors.length;
      colors[i * 3] = clusterColors[colorIndex][0];
      colors[i * 3 + 1] = clusterColors[colorIndex][1];
      colors[i * 3 + 2] = clusterColors[colorIndex][2];
    });
    
    return { positions, colors };
  }, [embeddings]);
  
  // Animation effect for gentle rotation and highlighting
  useFrame((state) => {
    if (pointsRef.current) {
      // Only rotate if no cluster is selected
      if (selectedCluster === null) {
        pointsRef.current.rotation.y = state.clock.getElapsedTime() * 0.05;
      }
      
      // Pulse effect when a cluster is selected
      if (selectedCluster !== null) {
        const scale = 1.0 + Math.sin(state.clock.getElapsedTime() * 3) * 0.05;
        pointsRef.current.scale.set(scale, scale, scale);
      } else {
        pointsRef.current.scale.set(1, 1, 1);
      }
    }
  });
  
  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attachObject={['attributes', 'position']}
          array={positions}
          count={positions.length / 3}
          itemSize={3}
        />
        <bufferAttribute
          attachObject={['attributes', 'color']}
          array={colors}
          count={colors.length / 3}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={selectedCluster !== null ? 0.2 : 0.15}
        vertexColors
        opacity={0.8}
        transparent
        alphaTest={0.5}
        sizeAttenuation
      />
    </points>
  );
}

export default GameEmbeddingsVisualization;
