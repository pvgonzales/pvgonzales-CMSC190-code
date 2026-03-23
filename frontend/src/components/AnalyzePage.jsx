import { useState, useEffect, useCallback } from 'react';
import Webcam from './Webcam';
import UploadVideo from './UploadVideo';
import Prediction from './Prediction';

const API_URL = import.meta.env.VITE_API_URL;

export default function AnalyzePage() {
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedModel, setSelectedModel] = useState('byol');
  const [availableModels, setAvailableModels] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch(`${API_URL}/api/models`);
        const data = await res.json();
        setAvailableModels(data.models);
        setBackendStatus('connected');
        if (data.models.length > 0) {
          setSelectedModel(data.models[0].id);
        }
      } catch {
        setBackendStatus('disconnected');
      }
    };
    checkBackend();

    const interval = setInterval(async () => {
      try {
        await fetch(`${API_URL}/api/health`);
        setBackendStatus('connected');
      } catch {
        setBackendStatus('disconnected');
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handlePrediction = useCallback((pred) => {
    setPrediction(pred);
  }, []);

  return (
    <div className="analyze-page">
      <div className="analyze-controls">
        <div className="analyze-tabs">
          <button
            className={`analyze-tab ${activeTab === 'webcam' ? 'active' : ''}`}
            onClick={() => { setActiveTab('webcam'); setPrediction(null); }}
          >
            Live Camera
          </button>
          <button
            className={`analyze-tab ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => { setActiveTab('upload'); setPrediction(null); }}
          >
            Upload Video
          </button>
        </div>

        <div className="analyze-model-select">
          <span className="model-label">Model:</span>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {availableModels.map((m) => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
            {availableModels.length === 0 && (
              <>
                <option value="byol">BYOL (3DResNet-18)</option>
                <option value="dino">DINOv3 (ViT)</option>
              </>
            )}
          </select>
        </div>
      </div>

      {backendStatus === 'disconnected' && (
        <div className="backend-banner">
          <span>
            Backend server is not running. Start it with:{' '}
            <code>cd backend && uvicorn main:app --reload --port 8000</code>
          </span>
        </div>
      )}

      <div className="analyze-content">
        <div className="analyze-main">
          {activeTab === 'webcam' ? (
            <Webcam
              selectedModel={selectedModel}
              onPrediction={handlePrediction}
            />
          ) : (
            <UploadVideo
              selectedModel={selectedModel}
              onPrediction={handlePrediction}
            />
          )}
        </div>

        <div className="analyze-sidebar">
          <Prediction prediction={prediction} />
        </div>
      </div>
    </div>
  );
}
