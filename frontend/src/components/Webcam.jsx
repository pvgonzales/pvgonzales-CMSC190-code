import { useState, useRef, useEffect, useCallback } from 'react';
import camera from '../assets/camera.png';

const API_URL = import.meta.env.VITE_API_URL;
const WS_URL = API_URL.replace(/^http/, 'ws');
const CAPTURE_FPS = 5;

export default function Webcam({ selectedModel, onPrediction }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const captureIntervalRef = useRef(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [bufferStatus, setBufferStatus] = useState({ buffered: 0, needed: 16 });
  const [error, setError] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);

  const startWebcam = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const ws = new WebSocket(`${WS_URL}/ws/realtime`);

      ws.onopen = () => {
        setWsConnected(true);
        ws.send(JSON.stringify({ action: 'start', model: selectedModel }));
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'prediction') {
          onPrediction(msg);
        } else if (msg.type === 'buffer') {
          setBufferStatus({ buffered: msg.buffered, needed: msg.needed });
        }
      };

      ws.onclose = () => setWsConnected(false);
      ws.onerror = () => {
        setError('WebSocket connection failed. Is the backend running?');
        setWsConnected(false);
      };

      wsRef.current = ws;

      captureIntervalRef.current = setInterval(() => {
        captureAndSendFrame();
      }, 1000 / CAPTURE_FPS);

      setIsStreaming(true);
    } catch (err) {
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied.');
      } else if (err.name === 'NotFoundError') {
        setError('No camera found.');
      } else {
        setError(`Camera error: ${err.message}`);
      }
    }
  }, [selectedModel, onPrediction]);

  const stopWebcam = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.onerror = null;
      wsRef.current.onclose = null;

      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: 'stop' }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }

    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
    
    setIsStreaming(false);
    setWsConnected(false);
    setBufferStatus({ buffered: 0, needed: 16 });
    
    setError(null); 
  }, []);

  const captureAndSendFrame = useCallback(() => {
    if (!videoRef.current || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;

    const video = videoRef.current;
    if (video.readyState < 2) return; 

    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
      canvasRef.current.width = 224;
      canvasRef.current.height = 224;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(video, 0, 0, 224, 224);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    wsRef.current.send(JSON.stringify({ action: 'frame', data: dataUrl }));
  }, []);

  useEffect(() => {
    return () => stopWebcam();
  }, [stopWebcam]);

  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'start', model: selectedModel }));
    }
  }, [selectedModel]);

  const bufferPercent = (bufferStatus.buffered / bufferStatus.needed) * 100;

  return (
    <div className="webcam-card">
      <div className="webcam-viewport" onClick={!isStreaming ? startWebcam : undefined}>
        
        <video 
          ref={videoRef} 
          muted 
          playsInline 
          style={{ display: isStreaming ? 'block' : 'none', width: '100%', height: '100%' }} 
        />

        {isStreaming ? (
          <div className="live-badge">
            <span className="rec-dot" />
            LIVE
          </div>
        ) : (
          <div className="webcam-placeholder-zone" onClick={startWebcam}>
            <img src={camera} alt="camera"/>
            <p className="upload-text">Click to Open Camera</p>
            {error && <p className="webcam-error">{error}</p>}
        </div>
        )}
      </div>

      {isStreaming && (
        <div className="webcam-status-bar">
          <span>Buffer: {bufferStatus.buffered}/{bufferStatus.needed} frames</span>
          <button className="btn-stop" onClick={stopWebcam}>⏹ Stop</button>
        </div>
      )}
    </div>
  );
}