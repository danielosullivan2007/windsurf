import React, { useState, useEffect } from 'react';
import * as d3 from 'd3';

interface GameEmbedding {
  title: string;
  cluster: number;
  tsneX: number;
  tsneY: number;
  tsneZ: number;
}

const CLUSTER_COLORS = [
  '#ff0000', // Red
  '#00ff00', // Green
  '#0000ff', // Blue
  '#ffff00', // Yellow
  '#ff00ff', // Magenta
  '#00ffff', // Cyan
  '#ffa500', // Orange
  '#800080', // Purple
  '#008000', // Dark Green
];

interface GameEmbeddingsVisualizationProps {
  embeddings: GameEmbedding[];
}

const GameEmbeddingsVisualization: React.FC<GameEmbeddingsVisualizationProps> = ({ embeddings: propEmbeddings }) => {
  const [embeddings, setEmbeddings] = useState<GameEmbedding[]>(propEmbeddings);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);

  useEffect(() => {
    // Ensure embeddings are available
    if (!propEmbeddings || propEmbeddings.length === 0) return;

    // Clear previous visualization
    d3.select('#scatter-plot-container').selectAll('*').remove();

    // Set up SVG
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const width = 800 - margin.left - margin.right;
    const height = 600 - margin.top - margin.bottom;

    const svg = d3.select('#scatter-plot-container')
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scale for x and y
    const xScale = d3.scaleLinear()
      .domain([d3.min(embeddings, d => d.tsneX) || 0, d3.max(embeddings, d => d.tsneX) || 1])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([d3.min(embeddings, d => d.tsneY) || 0, d3.max(embeddings, d => d.tsneY) || 1])
      .range([height, 0]);

    // Add points
    svg.selectAll('.point')
      .data(embeddings.filter(d => selectedCluster === null || d.cluster === selectedCluster))
      .enter()
      .append('circle')
      .attr('class', 'point')
      .attr('cx', d => xScale(d.tsneX))
      .attr('cy', d => yScale(d.tsneY))
      .attr('r', 5)
      .attr('fill', d => CLUSTER_COLORS[d.cluster % CLUSTER_COLORS.length])
      .on('mouseover', (event, d) => {
        d3.select('#tooltip')
          .style('display', 'block')
          .style('left', `${event.pageX + 10}px`)
          .style('top', `${event.pageY - 10}px`)
          .html(`
            <strong>Title:</strong> ${d.title}<br>
            <strong>Cluster:</strong> ${d.cluster}
          `);
      })
      .on('mouseout', () => {
        d3.select('#tooltip').style('display', 'none');
      });

    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    svg.append('g')
      .call(d3.axisLeft(yScale));

  }, [embeddings, selectedCluster]);

  return (
    <div className="game-embeddings-visualization">
      <h2>Game Embeddings t-SNE Visualization</h2>
      <div className="cluster-selector">
        <button onClick={() => setSelectedCluster(null)}>All Clusters</button>
        {CLUSTER_COLORS.map((_, index) => (
          <button 
            key={index} 
            onClick={() => setSelectedCluster(index)}
            style={{ backgroundColor: CLUSTER_COLORS[index], color: 'white' }}
          >
            Cluster {index}
          </button>
        ))}
      </div>
      <div 
        id="scatter-plot-container" 
        style={{ width: '800px', height: '600px' }}
      />
      <div 
        id="tooltip" 
        style={{
          position: 'absolute',
          background: 'rgba(0,0,0,0.7)',
          color: 'white',
          padding: '10px',
          borderRadius: '5px',
          display: 'none',
          pointerEvents: 'none',
        }}
      />
    </div>
  );
};

export default GameEmbeddingsVisualization;
