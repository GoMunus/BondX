import React, { useRef, useMemo, useState, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Line, Sphere } from '@react-three/drei'
import * as THREE from 'three'
import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown, Activity } from 'lucide-react'

interface YieldPoint {
  maturity: number // years
  yield: number // percentage
  liquidity: number // 0-1 score
  volume: number // trading volume
  spread: number // basis points
}

interface YieldCurve3DProps {
  data: YieldPoint[]
  shockValue?: number
  onPointHover?: (point: YieldPoint | null) => void
  className?: string
}

// 3D Yield Curve Component
const YieldCurveMesh: React.FC<{
  data: YieldPoint[]
  shockValue: number
  onPointHover: (point: YieldPoint | null) => void
}> = ({ data, shockValue, onPointHover }) => {
  const meshRef = useRef<THREE.Group>(null)
  const { camera, raycaster, mouse, scene } = useThree()
  const [hoveredPoint, setHoveredPoint] = useState<YieldPoint | null>(null)

  // Generate curve geometry
  const curveGeometry = useMemo(() => {
    const points = data.map((point, index) => {
      const x = point.maturity
      const y = point.yield + (shockValue / 100) // Apply shock
      const z = point.liquidity * 10 // Scale liquidity for 3D effect
      return new THREE.Vector3(x, y, z)
    })

    // Create smooth curve using CatmullRomCurve3
    const curve = new THREE.CatmullRomCurve3(points, false, 'centripetal')
    return curve.getPoints(100)
  }, [data, shockValue])

  // Generate point spheres
  const pointSpheres = useMemo(() => {
    return data.map((point, index) => {
      const x = point.maturity
      const y = point.yield + (shockValue / 100)
      const z = point.liquidity * 10
      
      // Color based on liquidity (green = liquid, red = illiquid)
      const color = new THREE.Color()
      color.setHSL(0.3 * point.liquidity, 0.8, 0.5)
      
      return {
        position: [x, y, z] as [number, number, number],
        color,
        size: Math.max(0.1, point.volume / 1000000), // Size based on volume
        data: point
      }
    })
  }, [data, shockValue])

  // Handle mouse interactions
  useFrame(() => {
    if (!meshRef.current) return

    raycaster.setFromCamera(mouse, camera)
    const intersects = raycaster.intersectObjects(scene.children, true)
    
    if (intersects.length > 0) {
      const intersect = intersects[0]
      const userData = intersect.object.userData
      if (userData && userData.pointData) {
        setHoveredPoint(userData.pointData)
        onPointHover(userData.pointData)
      }
    } else {
      setHoveredPoint(null)
      onPointHover(null)
    }
  })

  return (
    <group ref={meshRef}>
      {/* Yield curve line */}
      <Line
        points={curveGeometry}
        color="#00ff88"
        lineWidth={3}
        transparent
        opacity={0.8}
      />
      
      {/* Point spheres */}
      {pointSpheres.map((sphere, index) => (
        <Sphere
          key={index}
          position={sphere.position}
          args={[sphere.size, 16, 16]}
          userData={{ pointData: sphere.data }}
        >
          <meshStandardMaterial
            color={sphere.color}
            emissive={sphere.color}
            emissiveIntensity={0.2}
            transparent
            opacity={0.8}
          />
        </Sphere>
      ))}
      
      {/* Hovered point highlight */}
      {hoveredPoint && (
        <Sphere
          position={[
            hoveredPoint.maturity,
            hoveredPoint.yield + (shockValue / 100),
            hoveredPoint.liquidity * 10
          ]}
          args={[0.3, 16, 16]}
        >
          <meshStandardMaterial
            color="#ffffff"
            emissive="#ffffff"
            emissiveIntensity={0.5}
          />
        </Sphere>
      )}
    </group>
  )
}

// Main 3D Yield Curve Component
const YieldCurve3D: React.FC<YieldCurve3DProps> = ({
  data,
  shockValue = 0,
  onPointHover,
  className = ''
}) => {
  const [hoveredPoint, setHoveredPoint] = useState<YieldPoint | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Simulate data loading
  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 1000)
    return () => clearTimeout(timer)
  }, [])

  const handlePointHover = (point: YieldPoint | null) => {
    setHoveredPoint(point)
    onPointHover?.(point)
  }

  // Calculate curve statistics
  const curveStats = useMemo(() => {
    if (!data.length) return null
    
    const yields = data.map(p => p.yield + (shockValue / 100))
    const avgYield = yields.reduce((a, b) => a + b, 0) / yields.length
    const maxYield = Math.max(...yields)
    const minYield = Math.min(...yields)
    const spread = maxYield - minYield
    
    return { avgYield, maxYield, minYield, spread }
  }, [data, shockValue])

  if (isLoading) {
    return (
      <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-400">Loading 3D Yield Curve...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`bg-gray-900 rounded-lg overflow-hidden ${className}`}
    >
      {/* Header */}
      <div className="p-6 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-600 rounded-lg">
              <TrendingUp className="h-6 w-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white">3D Yield Curve</h3>
              <p className="text-gray-400 text-sm">Interactive yield curve visualization</p>
            </div>
          </div>
          
          {shockValue !== 0 && (
            <div className="flex items-center space-x-2 px-3 py-1 bg-red-600/20 rounded-full">
              <Activity className="h-4 w-4 text-red-400" />
              <span className="text-red-400 text-sm font-medium">
                +{shockValue}bps Shock Applied
              </span>
            </div>
          )}
        </div>
      </div>

      {/* 3D Canvas */}
      <div className="h-96 relative">
        <Canvas
          camera={{ position: [10, 5, 10], fov: 50 }}
          style={{ background: 'transparent' }}
        >
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <pointLight position={[-10, -10, -5]} intensity={0.5} />
          
          <YieldCurveMesh
            data={data}
            shockValue={shockValue}
            onPointHover={handlePointHover}
          />
          
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={5}
            maxDistance={20}
          />
        </Canvas>

        {/* Overlay Info */}
        <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-sm rounded-lg p-3">
          <div className="text-white text-sm space-y-1">
            <div>X: Maturity (Years)</div>
            <div>Y: Yield (%)</div>
            <div>Z: Liquidity Score</div>
          </div>
        </div>

        {/* Point Details */}
        {hoveredPoint && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="absolute bottom-4 right-4 bg-black/80 backdrop-blur-sm rounded-lg p-4 min-w-[200px]"
          >
            <h4 className="text-white font-medium mb-2">Bond Details</h4>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Maturity:</span>
                <span className="text-white">{hoveredPoint.maturity}Y</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Yield:</span>
                <span className="text-white">{(hoveredPoint.yield + shockValue/100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Liquidity:</span>
                <span className="text-white">{(hoveredPoint.liquidity * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Volume:</span>
                <span className="text-white">₹{(hoveredPoint.volume / 100000).toFixed(0)}L</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Spread:</span>
                <span className="text-white">{hoveredPoint.spread}bps</span>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Statistics */}
      {curveStats && (
        <div className="p-6 border-t border-gray-800">
          <div className="grid grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">
                {curveStats.avgYield.toFixed(2)}%
              </div>
              <div className="text-gray-400 text-sm">Avg Yield</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">
                {curveStats.maxYield.toFixed(2)}%
              </div>
              <div className="text-gray-400 text-sm">Max Yield</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-400">
                {curveStats.minYield.toFixed(2)}%
              </div>
              <div className="text-gray-400 text-sm">Min Yield</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">
                {curveStats.spread.toFixed(2)}%
              </div>
              <div className="text-gray-400 text-sm">Spread</div>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  )
}

export default YieldCurve3D
