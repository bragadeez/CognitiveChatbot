// components/chat/messages/VideoCard.tsx

import type { VideoData } from '../../../types';

interface Props { data: VideoData }

function getVideoId(url: string): string {
  const match = url.match(/v=([a-zA-Z0-9_-]+)/);
  return match?.[1] ?? '';
}

export default function VideoCard({ data }: Props) {
  const videoId = getVideoId(data.url);
  return (
    <div className="video-card">
      <div className="video-card-header">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--color-primary)', marginTop: 2, flexShrink: 0 }}>
          <polygon points="5 3 19 12 5 21 5 3" />
        </svg>
        <div className="video-info">
          <h4>{data.title}</h4>
          <div className="video-meta">
            {data.channel && (
              <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ marginRight: 4 }}>
                  <path d="M22.54 6.42a2.78 2.78 0 0 0-1.95-1.96C18.88 4 12 4 12 4s-6.88 0-8.59.46a2.78 2.78 0 0 0-1.95 1.96A29 29 0 0 0 1 12a29 29 0 0 0 .46 5.58 2.78 2.78 0 0 0 1.95 1.96C5.12 20 12 20 12 20s6.88 0 8.59-.46a2.78 2.78 0 0 0 1.95-1.96A29 29 0 0 0 23 12a29 29 0 0 0-.46-5.58z"/>
                  <polygon points="9.75 15.02 15.5 12 9.75 8.98 9.75 15.02"/>
                </svg>
                {data.channel}
              </span>
            )}
            {data.duration && (
              <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ marginRight: 4 }}>
                  <circle cx="12" cy="12" r="10" />
                  <polyline points="12 6 12 12 16 14" />
                </svg>
                {data.duration}
              </span>
            )}
            {data.views && (
              <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ marginRight: 4 }}>
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
                {data.views}
              </span>
            )}
          </div>
        </div>
      </div>
      {videoId && (
        <div className="video-embed">
          <iframe
            src={`https://www.youtube.com/embed/${videoId}`}
            title={data.title}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </div>
      )}
      <div className="video-card-footer">
        <a href={data.url} target="_blank" rel="noreferrer">
          Watch on YouTube →
        </a>
      </div>
    </div>
  );
}
