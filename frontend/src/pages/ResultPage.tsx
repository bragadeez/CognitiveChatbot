// pages/ResultPage.tsx — Learning style result with confidence ring and mode selection

import { useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import type { LearningStyle, VisualMode } from '../types';
import '../styles/result.css';

const STYLE_META: Record<LearningStyle, { description: string }> = {
  Visual: {
    description:
      'You learn best through diagrams, charts, and videos. Visual representations help you grasp complex ideas quickly.',
  },
  Auditory: {
    description:
      'You absorb information most effectively through listening, discussion, and spoken explanations.',
  },
  'Reading/Writing': {
    description:
      'You thrive with detailed written material, note-taking, and structured textual explanations.',
  },
  Kinesthetic: {
    description:
      'You learn by doing — through experiments, real-world examples, and hands-on practice.',
  },
};

export default function ResultPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const style = (params.get('style') ?? 'Visual') as LearningStyle;
  const confidence = parseFloat(params.get('confidence') ?? '0');

  const [showVisualOptions, setShowVisualOptions] = useState(false);
  const circleRef = useRef<SVGCircleElement>(null);

  const circumference = 2 * Math.PI * 45; // r=45

  useEffect(() => {
    if (circleRef.current) {
      const offset = circumference - (confidence / 100) * circumference;
      circleRef.current.style.strokeDashoffset = String(offset);
    }
  }, [confidence, circumference]);

  const handleContinue = () => {
    if (style === 'Visual') {
      setShowVisualOptions(true);
    } else {
      navigate(`/chat?style=${encodeURIComponent(style)}`);
    }
  };

  const handleVisualMode = (mode: VisualMode) => {
    navigate(`/chat?style=Visual&mode=${mode}`);
  };

  const meta = STYLE_META[style] ?? STYLE_META.Visual;

  return (
    <main className="result-page">
      <div className="blob-bg blob-1" />
      <div className="blob-bg blob-2" />
      <div className="result-card">
        <div className="result-badge">Assessment Complete</div>

        <p className="result-heading">Your Primary Learning Style</p>
        <div className="result-style">{style}</div>

        {/* Confidence ring */}
        <div className="confidence-ring">
          <svg viewBox="0 0 100 100" width="100" height="100">
            <defs>
              <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="var(--color-primary)" />
                <stop offset="100%" stopColor="var(--color-accent)" />
              </linearGradient>
            </defs>
            <circle className="track" cx="50" cy="50" r="45" />
            <circle
              ref={circleRef}
              className="fill"
              cx="50" cy="50" r="45"
              strokeDasharray={circumference}
              strokeDashoffset={circumference}
            />
          </svg>
          <div className="confidence-label">
            <span className="value">{Math.round(confidence)}</span>
            <span className="unit">% conf.</span>
          </div>
        </div>

        <p className="result-description">{meta.description}</p>

        {!showVisualOptions ? (
          <button className="cta-btn" id="btn-continue" onClick={handleContinue}>
            Start Learning →
          </button>
        ) : (
          <div className="visual-mode-section animate-fade-in-up">
            <p className="visual-mode-title">Choose your Visual Learning Mode:</p>
            <div className="mode-grid">
              <button className="mode-card" id="mode-image" onClick={() => handleVisualMode('image')}>
                <div className="icon">
                  <svg className="mode-icon-svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ margin: '0 auto', color: 'var(--color-primary)' }}>
                    <circle cx="12" cy="12" r="3"/>
                    <circle cx="6" cy="6" r="3"/>
                    <circle cx="18" cy="6" r="3"/>
                    <circle cx="6" cy="18" r="3"/>
                    <circle cx="18" cy="18" r="3"/>
                    <line x1="8.12" y1="8.12" x2="9.88" y2="9.88"/>
                    <line x1="15.88" y1="8.12" x2="14.12" y2="9.88"/>
                    <line x1="8.12" y1="15.88" x2="9.88" y2="14.12"/>
                    <line x1="15.88" y1="15.88" x2="14.12" y2="14.12"/>
                  </svg>
                </div>
                <div className="label">Mindmaps</div>
                <div className="desc">AI-generated concept diagrams</div>
              </button>
              <button className="mode-card" id="mode-video" onClick={() => handleVisualMode('video')}>
                <div className="icon">
                  <svg className="mode-icon-svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ margin: '0 auto', color: 'var(--color-accent)' }}>
                    <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/>
                    <line x1="7" y1="2" x2="7" y2="22"/>
                    <line x1="17" y1="2" x2="17" y2="22"/>
                    <line x1="2" y1="12" x2="22" y2="12"/>
                    <line x1="2" y1="7" x2="7" y2="7"/>
                    <line x1="2" y1="17" x2="7" y2="17"/>
                    <line x1="17" y1="17" x2="22" y2="17"/>
                    <line x1="17" y1="7" x2="22" y2="7"/>
                  </svg>
                </div>
                <div className="label">Videos</div>
                <div className="desc">Curated YouTube tutorials</div>
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
