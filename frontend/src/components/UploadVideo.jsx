import { useState, useRef, useCallback } from 'react';
import folder from '../assets/folder.png'

const API_URL = import.meta.env.VITE_API_URL;

export default function UploadVideo({ selectedModel, onPrediction, onResults }) {
  const fileInputRef = useRef(null);

  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFile = useCallback(async (selectedFile) => {
    if (!selectedFile) return;

    const validTypes = ['video/mp4', 'video/avi', 'video/webm', 'video/quicktime', 'video/x-msvideo'];
    if (!validTypes.includes(selectedFile.type) && !selectedFile.name.match(/\.(mp4|avi|webm|mov|mkv)$/i)) {
      setError('Please upload a video file (MP4, AVI, WebM, MOV)');
      return;
    }

    setFile(selectedFile);
    setError(null);
    setResults(null);
    setUploading(true);
    setProgress(10);
    setProgressText('Uploading video...');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      setProgress(30);
      setProgressText('Processing video on server...');

      const response = await fetch(
        `${API_URL}/api/upload-video?model=${selectedModel}&stride=8`,
        { method: 'POST', body: formData }
      );

      setProgress(80);
      setProgressText('Analyzing clips...');

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Upload failed');
      }

      const data = await response.json();
      setProgress(100);
      setProgressText('Analysis complete!');
      setResults(data);

      if (onResults) onResults(data);

      if (data.timeline && data.timeline.length > 0) {
        const dominantClass = data.summary.most_common_prediction;
        const bestClip = data.timeline
          .filter(t => t.prediction === dominantClass)
          .reduce((best, curr) =>
            curr.confidence > (best?.confidence || 0) ? curr : best
          , null);
        onPrediction(bestClip || data.timeline[0]);
      }
    } catch (err) {
      setError(err.message || 'Failed to analyze video');
      setProgress(0);
    } finally {
      setUploading(false);
    }
  }, [selectedModel, onPrediction, onResults]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [handleFile]);

  const handleDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => setDragOver(false);

  const reset = () => {
    setFile(null);
    setResults(null);
    setError(null);
    setProgress(0);
    setProgressText('');
    onPrediction(null);
  };

  const getClassColor = (className) => {
    const colors = {
      Normal: 'var(--color-normal)',
      Assault: 'var(--color-assault)',
      Abuse: 'var(--color-abuse)',
      Robbery: 'var(--color-robbery)',
      Shooting: 'var(--color-shooting)',
    };
    return colors[className] || 'var(--text-secondary)';
  };

  return (
    <div className="upload-card">
      {/* Upload Zone */}
      {!file && !results && (
        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className="upload-folder-icon">
            <img src={folder}/>
          </div>
          <p className="upload-text">Select or Drag and Drop a<br/>Surveillance Video</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>
      )}

      {/* Progress */}
      {uploading && (
        <div className="upload-progress fade-in">
          <p className="upload-filename">{file?.name}</p>
          <div className="progress-bar-container">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
          </div>
          <p className="progress-text">{progressText}</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="upload-error fade-in">
            {error}
        </div>
      )}

      {/* Results */}
      {results && !uploading && (
        <div className="upload-results fade-in">
          {/* File info bar */}
          <div className="results-info-bar">
            <span>File: &lt;{results.video_info.filename}&gt;</span>
            <span>Duration: {results.video_info.duration_str}</span>
            <button className="btn-close" onClick={reset}>✕ Close</button>
          </div>

          {/* Detection per Frame */}
          <div className="results-timeline">
            <h4 className="timeline-heading">Detection per Frame</h4>
            <div className="timeline-list">
              {results.timeline.map((item, i) => (
                <div
                  key={i}
                  className="timeline-row"
                  style={{ borderLeftColor: getClassColor(item.prediction) }}
                  onClick={() => onPrediction(item)}
                >
                  <span className="timeline-time">{item.timestamp_str}</span>
                  <span className="timeline-label" style={{ color: getClassColor(item.prediction) }}>
                    {item.prediction}
                  </span>
                  <span className="timeline-conf">{(item.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
