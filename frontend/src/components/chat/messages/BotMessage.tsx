// components/chat/messages/BotMessage.tsx — Renders all bot message types

import { marked } from 'marked';
import type { ChatMessage } from '../../../types';
import VideoCard from './VideoCard';
import MindmapCard from './MindmapCard';
import ResourceCard from './ResourceCard';

marked.setOptions({ breaks: true });

interface Props { message: ChatMessage }

export default function BotMessage({ message }: Props) {
  if ((message.type as string) === 'loading') {
    return (
      <div className="message-row">
        <div className="message-avatar bot">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'white' }}>
            <path d="M12 2a3 3 0 0 0-3 3v2a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
            <rect x="4" y="7" width="16" height="13" rx="2"/>
            <path d="M9 12h.01M15 12h.01M8 16h8"/>
          </svg>
        </div>
        <div className="message-bubble bot">
          <div className="typing-indicator">
            <span className="typing-dot" />
            <span className="typing-dot" />
            <span className="typing-dot" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="message-row animate-fade-in-up">
      <div className="message-avatar bot">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'white' }}>
          <path d="M12 2a3 3 0 0 0-3 3v2a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
          <rect x="4" y="7" width="16" height="13" rx="2"/>
          <path d="M9 12h.01M15 12h.01M8 16h8"/>
        </svg>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxWidth: '78%' }}>

        {/* Text content (shown for audio/resource modes alongside their media) */}
        {message.content && message.type !== 'loading' && (
          <div
            className="message-bubble bot"
            dangerouslySetInnerHTML={{ __html: marked.parse(message.content) as string }}
          />
        )}

        {/* Video embed */}
        {message.type === 'video' && message.videoData && (
          <VideoCard data={message.videoData} />
        )}

        {/* Mindmap diagram */}
        {message.type === 'diagram' && message.diagramData && (
          <MindmapCard data={message.diagramData} />
        )}

        {/* Audio player */}
        {message.type === 'audio' && message.audioUrl && (
          <div className="audio-wrapper">
            <audio controls src={message.audioUrl} />
          </div>
        )}

        {/* Resource card (kinesthetic) */}
        {message.type === 'resource' && message.resourceData && (
          <ResourceCard data={message.resourceData} />
        )}
      </div>
    </div>
  );
}
