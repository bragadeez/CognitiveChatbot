// api/chatApi.ts — Typed fetch wrappers for chat and media routes

import type { LLMRequest, LLMResponse, VideoData } from '../types';

const BASE = '/api';

export async function sendChatMessage(body: LLMRequest): Promise<LLMResponse> {
  const res = await fetch(`${BASE}/llm_response`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`LLM request failed: ${res.status}`);
  return res.json();
}

export async function searchVideo(query: string): Promise<VideoData> {
  const res = await fetch(`${BASE}/get_video`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error(`Video search failed: ${res.status}`);
  return res.json();
}

export async function generateDiagram(
  prompt: string
): Promise<{ image: string; code: string; explanation: string }> {
  const res = await fetch(`${BASE}/generate_image`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error(`Diagram generation failed: ${res.status}`);
  return res.json();
}

export async function textToSpeech(
  text: string,
  session_id: string
): Promise<{ audio_url?: string; blob?: Blob }> {
  const res = await fetch(`${BASE}/text_to_speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, session_id }),
  });
  if (!res.ok) throw new Error(`TTS failed: ${res.status}`);

  const contentType = res.headers.get('Content-Type') || '';
  if (contentType.includes('application/json')) {
    return res.json();
  }
  // Streaming raw bytes
  return { blob: await res.blob() };
}
