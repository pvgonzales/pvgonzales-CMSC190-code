import { useState } from 'react';
import alerticon from '../assets/alert.svg';
import normalicon from '../assets/normal.png';
import awaiting from '../assets/awaiting.png';
import probicon from '../assets/probicon.png';

const CLASS_COLORS = {
  Normal: 'var(--color-normal)',
  Assault: 'var(--color-assault)',
  Abuse: 'var(--color-abuse)',
  Robbery: 'var(--color-robbery)',
  Shooting: 'var(--color-shooting)',
};

const CLASS_ORDER = ['Normal', 'Assault', 'Abuse', 'Robbery', 'Shooting'];

export default function Prediction({ prediction }) {
  if (!prediction) {
    return (
      <div className="prediction-panel">
        <div className="alert-card">
          <div className="alert-icon">
            <img src={awaiting} alt="Awaiting video" style={{ width: '48px', height: '48px', objectFit: 'contain' }} />
          </div>
          <div className="alert-label" style={{ color: 'var(--text-primary)' }}>
            Awaiting
          </div>
          <div className="alert-label" style={{ color: 'var(--text-primary)', marginTop: -4 }}>
            Video Input
          </div>
        </div>

        <div className="probabilities-card">
          <div className="prob-header">
            <img src={probicon} alt="Probabilities" style={{ width: '18px', height: '18px', objectFit: 'contain' }} />
            <span>Class Probabilities</span>
          </div>
          <div className="prob-list">
            {CLASS_ORDER.map((cls) => (
              <div key={cls} className="prob-row">
                <div className="prob-row-header">
                  <span className="prob-label">
                    <span className="prob-dot" style={{ background: CLASS_COLORS[cls] }} />
                    {cls}
                  </span>
                  <span className="prob-value">0.0 %</span>
                </div>
                <div className="prob-bar-track">
                  <div className="prob-bar-fill" style={{ width: '0%', background: CLASS_COLORS[cls] }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const {
    prediction: predClass,
    confidence,
    class_scores: classScores,
  } = prediction;

  const alertBorderColor = CLASS_COLORS[predClass] || 'var(--text-secondary)';

  return (
    <div className="prediction-panel slide-in">
      <div
        className="alert-card"
        style={{ 
          borderColor: alertBorderColor,
          backgroundColor: `color-mix(in srgb, ${alertBorderColor} 15%, var(--bg-card))` 
        }}
      >
        <div className="alert-icon">
          {predClass === 'Normal' ? (
            <img 
              src={normalicon} 
              alt="Normal status" 
              style={{ width: '48px', height: '48px', objectFit: 'contain' }} 
            />
          ) : (
            <div 
              className="dynamic-svg-icon"
              style={{
                backgroundColor: alertBorderColor,
                WebkitMaskImage: `url(${alerticon})`,
                maskImage: `url(${alerticon})`
              }}
            />
          )}
        </div>
        <div className="alert-label" style={{ color: alertBorderColor }}>
          {predClass}
        </div>
        <div className="alert-confidence">
          {(confidence * 100).toFixed(1)}% confidence
        </div>
      </div>

      {/* Class Probabilities */}
      <div className="probabilities-card">
        <div className="prob-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--text-secondary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 8V6a2 2 0 0 1 2-2h2" />
            <path d="M4 16v2a2 2 0 0 0 2 2h2" />
            <path d="M16 4h2a2 2 0 0 1 2 2v2" />
            <path d="M16 20h2a2 2 0 0 0 2-2v-2" />
            <circle cx="12" cy="12" r="3" />
          </svg>
          <span>Class Probabilities</span>
        </div>
        <div className="prob-list">
          {CLASS_ORDER.map((cls) => {
            const score = classScores?.[cls] ?? 0;
            const pct = (score * 100).toFixed(1);

            return (
              <div key={cls} className="prob-row">
                <div className="prob-row-header">
                  <span className="prob-label">
                    <span className="prob-dot" style={{ background: CLASS_COLORS[cls] }} />
                    {cls}
                  </span>
                  <span className="prob-value">
                    {pct} %
                  </span>
                </div>
                <div className="prob-bar-track">
                  <div
                    className="prob-bar-fill"
                    style={{
                      width: `${score * 100}%`,
                      background: CLASS_COLORS[cls]
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}