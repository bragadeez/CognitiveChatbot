// types/index.ts — All shared TypeScript interfaces and types

export type LearningStyle = 'Visual' | 'Auditory' | 'Reading/Writing' | 'Kinesthetic';
export type VisualMode = 'image' | 'video';
export type MessageSender = 'user' | 'bot';
export type MessageType = 'text' | 'video' | 'diagram' | 'audio' | 'resource' | 'loading';

// ── Assessment ────────────────────────────────────────────────────────────────

export interface QuestionResponse {
  question?: string;
  index?: number;
  total?: number;
  done: boolean;
  style?: LearningStyle;
  confidence?: number;
}

export interface AnswerRequest {
  answer: 'Strongly Agree' | 'Agree' | 'Neutral' | 'Disagree' | 'Strongly Disagree';
  session_id: string;
}

export interface SubmitAnswerResponse {
  status?: string;
  done: boolean;
  style?: LearningStyle;
  confidence?: number;
}

export interface AssessmentResult {
  style: LearningStyle;
  confidence: number;
}

// ── Chat ──────────────────────────────────────────────────────────────────────

export interface VideoData {
  title: string;
  url: string;
  channel: string;
  duration: string;
  views: string;
}

export interface ResourceData {
  title: string;
  url: string;
  description: string;
}

export interface DiagramData {
  image: string;       // base64 data URI
  code: string;        // raw Mermaid code
  explanation: string;
}

export interface HistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatMessage {
  id: string;
  sender: MessageSender;
  type: MessageType;
  content?: string;
  videoData?: VideoData;
  resourceData?: ResourceData;
  diagramData?: DiagramData;
  audioUrl?: string;
}

export interface LLMRequest {
  query: string;
  style: LearningStyle;
  visual_type?: VisualMode;
  history: HistoryMessage[];
}

export interface LLMResponse {
  response?: string;
  audio?: string;
  image?: string;
  code?: string;
  explanation?: string;
  resource?: ResourceData;
  video_data?: VideoData;
  type: 'text' | 'video' | 'diagram' | 'audio' | 'resource';
  success: boolean;
}
