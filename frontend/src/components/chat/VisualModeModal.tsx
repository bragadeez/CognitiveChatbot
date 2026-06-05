// components/chat/VisualModeModal.tsx

import type { VisualMode } from '../../types';

interface Props {
  onSelect: (mode: VisualMode) => void;
  onClose: () => void;
}

export default function VisualModeModal({ onSelect, onClose }: Props) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={e => e.stopPropagation()}>
        <h3>Choose Visual Mode</h3>
        <p>How would you like to visualise ML concepts?</p>
        <div className="modal-options">
          <button className="modal-option" id="opt-image" onClick={() => onSelect('image')}>
            <span className="opt-icon">
              <svg className="opt-icon-svg" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--color-primary)' }}>
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
            </span>
            <span className="opt-label">Mindmaps</span>
          </button>
          <button className="modal-option" id="opt-video" onClick={() => onSelect('video')}>
            <span className="opt-icon">
              <svg className="opt-icon-svg" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--color-accent)' }}>
                <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/>
                <line x1="7" y1="2" x2="7" y2="22"/>
                <line x1="17" y1="2" x2="17" y2="22"/>
                <line x1="2" y1="12" x2="22" y2="12"/>
                <line x1="2" y1="7" x2="7" y2="7"/>
                <line x1="2" y1="17" x2="7" y2="17"/>
                <line x1="17" y1="17" x2="22" y2="17"/>
                <line x1="17" y1="7" x2="22" y2="7"/>
              </svg>
            </span>
            <span className="opt-label">Videos</span>
          </button>
        </div>
      </div>
    </div>
  );
}
