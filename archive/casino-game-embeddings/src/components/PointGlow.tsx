import * as THREE from 'three'
import React, { useRef } from 'react'
import { useFrame } from '@react-three/fiber'

interface GlowProps {
  position: [number, number, number]
  color: string
  scale?: number
}

export function PointGlow({ position, color, scale = 1.2 }: GlowProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  useFrame(({ clock }) => {
    if (meshRef.current) {
      const t = clock.getElapsedTime()
      meshRef.current.scale.setScalar(scale + Math.sin(t * 2) * 0.05)
    }
  })

  return (
    <mesh position={position} ref={meshRef}>
      <sphereGeometry args={[0.8, 32, 32]} />
      <meshBasicMaterial
        color={color}
        transparent={true}
        opacity={0.5}
        side={THREE.FrontSide}
        toneMapped={false}
      />
    </mesh>
  )
}
