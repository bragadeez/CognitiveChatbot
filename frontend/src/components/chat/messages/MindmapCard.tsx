// components/chat/messages/MindmapCard.tsx

import type { DiagramData } from '../../../types';

interface Props { data: DiagramData }

export default function MindmapCard({ data }: Props) {
  return (
    <div className="mindmap-card">
      <img src={data.image} alt="Mindmap diagram" />
      {data.explanation && (
        <div className="mindmap-explanation">
          <svg className="mindmap-concept-svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ verticalAlign: 'text-top', marginRight: 4, color: 'var(--color-accent)' }}>
            <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A5.5 5.5 0 0 0 7 8c0 1.3.5 2.6 1.5 3.5.7.8 1.3 1.5 1.5 2.5" />
            <path d="M9 18h6M10 22h4" />
          </svg>
          <strong>Concept:</strong> {data.explanation}
        </div>
      )}
    </div>
  );
}
