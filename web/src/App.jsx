import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const canvasRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    
    // Initialize canvas
    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    // Drawing settings
    ctx.strokeStyle = 'white'
    ctx.lineWidth = 25 // 增加笔画宽度，使其更接近 MNIST/EMNIST 的笔触
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
  }, [])

  const startDrawing = (e) => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const rect = canvas.getBoundingClientRect()
    
    const clientX = e.clientX || e.touches[0].clientX
    const clientY = e.clientY || e.touches[0].clientY
    
    const x = clientX - rect.left
    const y = clientY - rect.top
    
    ctx.beginPath()
    ctx.moveTo(x, y)
    setIsDrawing(true)
  }

  const draw = (e) => {
    if (!isDrawing) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const rect = canvas.getBoundingClientRect()
    
    const clientX = e.clientX || e.touches?.[0]?.clientX || e.clientX
    const clientY = e.clientY || e.touches?.[0]?.clientY || e.clientY
    
    const x = clientX - rect.left
    const y = clientY - rect.top
    
    ctx.lineTo(x, y)
    ctx.stroke()
  }

  const stopDrawing = () => {
    setIsDrawing(false)
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    setPrediction(null)
    setConfidence(null)
    setError(null)
  }

  const predictDigit = async () => {
    const canvas = canvasRef.current
    const imageData = canvas.toDataURL('image/png')
    
    setIsLoading(true)
    setError(null)
    
    try {
      // Use relative path to leverage Vite proxy and avoid CORS issues
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      })

      if (!response.ok) {
        throw new Error('Prediction request failed')
      }

      const result = await response.json()
      setPrediction(result.digit)
      setConfidence(result.confidence)
    } catch (err) {
      setError('Failed to recognize digit')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>MNIST Recognition</h1>
      <div className="card">
        <div className="canvas-wrapper">
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseOut={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
            style={{ touchAction: 'none' }}
          />
        </div>
        
        <div className="controls">
          <button onClick={clearCanvas} disabled={isLoading}>
            Clear
          </button>
          <button onClick={predictDigit} disabled={isLoading}>
            {isLoading ? 'Processing...' : 'Recognize'}
          </button>
        </div>

        {(prediction !== null || error) && (
          <div className="result">
            {error ? (
              <p className="error">{error}</p>
            ) : (
              <>
                <p className="prediction-text">
                  Prediction: <strong>{prediction}</strong>
                </p>
                <p className="confidence-text">
                  Confidence: {(confidence * 100).toFixed(2)}%
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
