import React, { useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import './GameVisualization.css';

// A simple rotating cube component
function Cube(props) {
  const meshRef = useRef();
  const [hovered, setHover] = useState(false);
  const [active, setActive] = useState(false);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.x += 0.01;
      meshRef.current.rotation.y += 0.01;
    }
  });

  return (
    <mesh
      {...props}
      ref={meshRef}
      scale={active ? 1.5 : 1}
      onClick={() => setActive(!active)}
      onPointerOver={() => setHover(true)}
      onPointerOut={() => setHover(false)}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color={hovered ? 'hotpink' : 'orange'} />
    </mesh>
  );
}

// Search component
function SearchBar({ onSearch }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(query);
  };

  return (
    <div className="search-container">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search for casino games..."
          className="search-input"
        />
        <button type="submit" className="search-button">Search</button>
      </form>
    </div>
  );
}

// Main Game Visualization component
function GameVisualization() {
  const [searchResults, setSearchResults] = useState([]);

  const handleSearch = (query) => {
    // In a real app, this would call an API or search a dataset
    console.log(`Searching for: ${query}`);
    // Simulating search results
    setSearchResults([
      { id: 1, name: 'Poker', popularity: 0.9 },
      { id: 2, name: 'Blackjack', popularity: 0.8 },
      { id: 3, name: 'Roulette', popularity: 0.7 }
    ]);
  };

  return (
    <div className="game-visualization">
      <h1>Casino Game Viewer</h1>
      <SearchBar onSearch={handleSearch} />
      
      {searchResults.length > 0 && (
        <div className="search-results">
          <h2>Search Results</h2>
          <ul>
            {searchResults.map(game => (
              <li key={game.id}>
                {game.name} - Popularity: {game.popularity}
              </li>
            ))}
          </ul>
        </div>
      )}
      
      <div className="visualization-container" style={{ height: '500px' }}>
        <Canvas>
          <ambientLight intensity={0.5} />
          <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} />
          <Cube position={[-1.2, 0, 0]} />
          <Cube position={[1.2, 0, 0]} />
          <OrbitControls />
        </Canvas>
      </div>
    </div>
  );
}

export default GameVisualization;
