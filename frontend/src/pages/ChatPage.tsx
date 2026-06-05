// pages/ChatPage.tsx — Full chatbot interface with all learning modes

import { useRef, useEffect, useState, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useChat } from '../hooks/useChat';
import BotMessage from '../components/chat/messages/BotMessage';
import VisualModeModal from '../components/chat/VisualModeModal';
import type { LearningStyle, VisualMode } from '../types';
import '../styles/chat.css';

const STYLE_SVGS: Record<LearningStyle, React.ReactNode> = {
  Visual: (
    <svg className="badge-svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  ),
  Auditory: (
    <svg className="badge-svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 18v-6a9 9 0 0 1 18 0v6" />
      <path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z" />
    </svg>
  ),
  'Reading/Writing': (
    <svg className="badge-svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  ),
  Kinesthetic: (
    <svg className="badge-svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  ),
};

const STYLE_WELCOME: Record<LearningStyle, string> = {
  Visual:            'Ask me any ML concept and I\'ll generate a mindmap or find a video for you!',
  Auditory:          'Ask me anything and I\'ll give you a clear spoken-style explanation with audio.',
  'Reading/Writing': 'Ask me anything and I\'ll give you a detailed, structured written explanation.',
  Kinesthetic:       'Ask me anything and I\'ll explain it with real-world analogies and hands-on examples.',
};

function getSessionId(): string {
  let id = sessionStorage.getItem('cog_session_id');
  if (!id) { id = crypto.randomUUID(); sessionStorage.setItem('cog_session_id', id); }
  return id;
}

export default function ChatPage() {
  const [params] = useSearchParams();
  const initialStyle = (params.get('style') ?? 'Visual') as LearningStyle;
  const initialMode  = (params.get('mode')  ?? 'image') as VisualMode;
  const sessionId    = getSessionId();

  const { messages, style, visualMode, isSending, sendMessage, changeStyle, setVisualMode } =
    useChat(initialStyle, sessionId);

  const [input, setInput]           = useState('');
  const [menuOpen, setMenuOpen]     = useState(false);
  const [modalOpen, setModalOpen]   = useState(false);
  const bodyRef   = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLInputElement>(null);

  // Style change toast state
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [toastVisible, setToastVisible] = useState(false);
  const prevStyleRef = useRef<LearningStyle>(style);
  const prevModeRef = useRef<VisualMode>(visualMode);

  // Set initial visual mode from URL param
  useEffect(() => { setVisualMode(initialMode); }, [initialMode, setVisualMode]);

  // Toast notifier for learning style changes
  useEffect(() => {
    const styleChanged = prevStyleRef.current !== style;
    const modeChanged = style === 'Visual' && prevModeRef.current !== visualMode;

    if (styleChanged || modeChanged) {
      let desc = '';
      if (style === 'Visual') {
        desc = visualMode === 'image'
          ? 'Visual Mode: Explanations focused on mindmaps & diagrams.'
          : 'Visual Mode: Explanations focused on educational videos.';
      } else if (style === 'Auditory') {
        desc = 'Auditory Mode: Explanations optimized for text-to-speech listening.';
      } else if (style === 'Reading/Writing') {
        desc = 'Reading & Writing Mode: Explanations presented as structured documents.';
      } else if (style === 'Kinesthetic') {
        desc = 'Kinesthetic Mode: Explanations focused on real-world analogies and practice.';
      }

      setToastMessage(desc);
      setToastVisible(true);

      const timer = setTimeout(() => {
        setToastVisible(false);
      }, 4000);

      prevStyleRef.current = style;
      prevModeRef.current = visualMode;

      return () => clearTimeout(timer);
    }
  }, [style, visualMode]);

  // Auto-scroll to bottom
  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const handleSend = useCallback(() => {
    if (!input.trim() || isSending) return;
    sendMessage(input.trim());
    setInput('');
    inputRef.current?.focus();
  }, [input, isSending, sendMessage]);

  const handleStyleChange = (newStyle: LearningStyle) => {
    setMenuOpen(false);
    if (newStyle === 'Visual') {
      setModalOpen(true);
    } else {
      changeStyle(newStyle);
    }
  };

  const handleVisualModeSelect = (mode: VisualMode) => {
    setModalOpen(false);
    changeStyle('Visual', mode);
  };

  return (
    <div className="chat-page">
      <div className="blob-bg blob-1" />
      <div className="blob-bg blob-2" />
      {/* Toast Switch Notification */}
      {toastVisible && toastMessage && (
        <div className="style-switch-toast">
          <div className="toast-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 6L9 17l-5-5"/>
            </svg>
          </div>
          <div className="toast-text">{toastMessage}</div>
        </div>
      )}

      {/* Header */}
      <header className="chat-header">
        <div className="chat-header-left">
          <div className="chat-logo">
            <svg className="chat-header-logo-svg" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--color-primary)' }}>
              <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-4.12 2.5 2.5 0 0 1 0-4.88 2.5 2.5 0 0 1 2.46-3.06zM14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-4.12 2.5 2.5 0 0 0 0-4.88A2.5 2.5 0 0 0 14.5 2z"/>
            </svg>
          </div>
          <div>
            <div className="chat-title">Cognitive Chatbot</div>
            <div className="chat-subtitle">Personalised ML Tutor</div>
          </div>
        </div>

        <div className="header-menu-container" style={{ position: 'relative' }}>
          <div className="style-badge" onClick={() => setMenuOpen(o => !o)}>
            {STYLE_SVGS[style]}
            <span className="badge-label">{style}</span>
            {style === 'Visual' && (
              <span className="badge-visual-mode">
                ({visualMode === 'image' ? 'Mindmap' : 'Video'})
              </span>
            )}
            <span className="badge-arrow">▾</span>
          </div>

          {menuOpen && (
            <div className="menu-dropdown">
              {(['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic'] as LearningStyle[]).map(s => (
                <button key={s} onClick={() => handleStyleChange(s)} className={s === style ? 'active' : ''}>
                  {STYLE_SVGS[s]}
                  <span>{s}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </header>

      {/* Body */}
      <div className="chat-body" ref={bodyRef}>
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <div className="welcome-icon" style={{ color: 'var(--color-primary)' }}>
              {STYLE_SVGS[style]}
            </div>
            <h2>Hello, I'm your {style} Tutor</h2>
            <p>{STYLE_WELCOME[style]}</p>
          </div>
        ) : (
          messages.map((msg) =>
            msg.sender === 'user' ? (
              <div key={msg.id} className="message-row user animate-fade-in-up">
                <div className="message-avatar user">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--text-secondary)' }}>
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                  </svg>
                </div>
                <div className="message-bubble user">{msg.content}</div>
              </div>
            ) : (
              <BotMessage key={msg.id} message={msg} />
            )
          )
        )}
      </div>

      {/* Input */}
      <div className="chat-input-area">
        <div className="chat-input-row">
          <input
            ref={inputRef}
            className="chat-input-box"
            id="chat-input"
            type="text"
            placeholder={`Ask about any ML concept…`}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            disabled={isSending}
          />
          <button
            className="send-btn"
            id="btn-send"
            onClick={handleSend}
            disabled={isSending || !input.trim()}
            title="Send"
          >
            {isSending ? (
              <span className="spinner-icon" />
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Visual mode modal */}
      {modalOpen && (
        <VisualModeModal
          onSelect={handleVisualModeSelect}
          onClose={() => setModalOpen(false)}
        />
      )}
    </div>
  );
}
