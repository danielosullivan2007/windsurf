import React, { useState, useEffect, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import './App.css';

// Component to display the 3D visualization
function GameVisualization({ data }) {
  // Generate colors for each cluster
  const generateClusterColors = () => {
    const numClusters = Math.max(...data.map(item => item.cluster)) + 1;
    return Array.from({ length: numClusters }, (_, i) => {
      // Generate vibrant, distinct colors for each cluster
      return [
        Math.sin(i * 0.7) * 0.5 + 0.5,
        Math.sin(i * 0.3 + 2) * 0.5 + 0.5,
        Math.sin(i * 0.5 + 4) * 0.5 + 0.5
      ];
    });
  };

  const clusterColors = generateClusterColors();

  // Find min/max values for scaling
  const findMinMax = (data, key) => {
    const values = data.map(item => item[key]);
    return {
      min: Math.min(...values),
      max: Math.max(...values)
    };
  };

  const xScale = findMinMax(data, 'tsneX');
  const yScale = findMinMax(data, 'tsneY');
  const zScale = findMinMax(data, 'tsneZ');

  // Normalize coordinates to [-15, 15] range
  const normalize = (value, min, max) => {
    return ((value - min) / (max - min) * 30) - 15;
  };

  // Points component using instanced mesh for better performance
  const Points = ({ data }) => {
    const [hoveredPoint, setHoveredPoint] = useState(null);
    const [selectedPoint, setSelectedPoint] = useState(null);

    return (
      <>
        {data.map((point, index) => {
          const x = normalize(point.tsneX, xScale.min, xScale.max);
          const y = normalize(point.tsneY, yScale.min, yScale.max);
          const z = normalize(point.tsneZ, zScale.min, zScale.max);
          const color = clusterColors[point.cluster];
          
          return (
            <mesh
              key={index}
              position={[x, y, z]}
              onPointerOver={() => setHoveredPoint(point)}
              onPointerOut={() => setHoveredPoint(null)}
              onClick={() => setSelectedPoint(point)}
            >
              <sphereGeometry args={[0.2, 16, 16]} />
              <meshBasicMaterial 
                color={hoveredPoint === point ? 'white' : `rgb(${color[0] * 255}, ${color[1] * 255}, ${color[2] * 255})`}
                toneMapped={false}
                emissive={`rgb(${color[0] * 255}, ${color[1] * 255}, ${color[2] * 255})`}
                emissiveIntensity={0.5}
              />
            </mesh>
          );
        })}

        {/* Display info for hovered/selected point */}
        {(hoveredPoint || selectedPoint) && (
          <div
            className="point-info"
            style={{
              position: 'absolute',
              bottom: '20px',
              left: '20px',
              color: 'white',
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              padding: '10px',
              borderRadius: '5px',
              maxWidth: '400px',
              zIndex: 100
            }}
          >
            <h3>{hoveredPoint?.title || selectedPoint?.title}</h3>
            <p><strong>Provider:</strong> {hoveredPoint?.provider || selectedPoint?.provider}</p>
            <p><strong>Cluster:</strong> {hoveredPoint?.cluster || selectedPoint?.cluster}</p>
            <p><strong>Summary:</strong> {hoveredPoint?.summary || selectedPoint?.summary}</p>
          </div>
        )}
      </>
    );
  };

  return (
    <group>
      {/* Add axes for reference */}
      <axesHelper args={[20]} />
      
      {/* Add the data points */}
      <Points data={data} />
    </group>
  );
}

function App() {
  const [gameData, setGameData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/data/game_summary_embeddings_tsne.json')
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to load data');
        }
        return response.json();
      })
      .then(data => {
        setGameData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading data:', err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Casino Game Clusters Visualization</h1>
        <p>Interactive 3D visualization of casino game clusters based on structured summaries</p>
      </header>

      <main>
        {loading ? (
          <div className="loading">Loading game data...</div>
        ) : error ? (
          <div className="error">Error: {error}</div>
        ) : (
          <div className="visualization-container">
            <Canvas
              camera={{ position: [0, 0, 30], fov: 50 }}
              style={{ height: '80vh', background: '#111' }}
            >
              <ambientLight intensity={0.5} />
              <pointLight position={[10, 10, 10]} intensity={1} />
              <Suspense fallback={null}>
                <GameVisualization data={gameData} />
              </Suspense>
              <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
            </Canvas>

            <div className="legend">
              <h3>Cluster Legend</h3>
              <div className="cluster-colors">
                {Array.from({ length: Math.max(...gameData.map(item => item.cluster)) + 1 }, (_, i) => {
                  const color = [
                    Math.sin(i * 0.7) * 0.5 + 0.5,
                    Math.sin(i * 0.3 + 2) * 0.5 + 0.5,
                    Math.sin(i * 0.5 + 4) * 0.5 + 0.5
                  ];
                  return (
                    <div key={i} className="cluster-color">
                      <div 
                        className="color-box" 
                        style={{ 
                          backgroundColor: `rgb(${color[0] * 255}, ${color[1] * 255}, ${color[2] * 255})`,
                          width: '20px',
                          height: '20px',
                          display: 'inline-block',
                          marginRight: '10px',
                          borderRadius: '3px'
                        }} 
                      />
                      <span>Cluster {i}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </main>

      <footer>
        <p>Created with React and Three.js | Casino Game Embeddings Project</p>
      </footer>
    </div>
  );
}

export default App;
