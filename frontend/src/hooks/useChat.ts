// hooks/useChat.ts — Chat history state management with context preservation.
// Maintains conversation history sent to the backend on every message,
// so the LLM has full multi-turn context. History resets on style change.

import { useState, useCallback } from 'react';
import { sendChatMessage, textToSpeech } from '../api/chatApi';
import type {
  ChatMessage,
  HistoryMessage,
  LearningStyle,
  VisualMode,
} from '../types';

const MAX_HISTORY_PAIRS = 10; // keep last 10 user+assistant pairs = 20 messages

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

export function useChat(initialStyle: LearningStyle, sessionId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [history, setHistory] = useState<HistoryMessage[]>([]);
  const [style, setStyle] = useState<LearningStyle>(initialStyle);
  const [visualMode, setVisualMode] = useState<VisualMode>('image');
  const [isSending, setIsSending] = useState(false);

  const addMessage = useCallback((msg: ChatMessage) => {
    setMessages(prev => [...prev, msg]);
  }, []);

  const changeStyle = useCallback((newStyle: LearningStyle, newVisualMode?: VisualMode) => {
    setStyle(newStyle);
    if (newVisualMode) setVisualMode(newVisualMode);
    // Reset history when style changes — new context for the new mode
    setHistory([]);
    addMessage({
      id: generateId(),
      sender: 'bot',
      type: 'text',
      content: `🔄 Switched to **${newStyle}** learning mode. Let's continue!`,
    });
  }, [addMessage]);

  const sendMessage = useCallback(async (query: string) => {
    if (!query.trim() || isSending) return;

    // Add user message
    const userMsg: ChatMessage = { id: generateId(), sender: 'user', type: 'text', content: query };
    addMessage(userMsg);
    setIsSending(true);

    // Add loading indicator
    const loadingId = generateId();
    addMessage({ id: loadingId, sender: 'bot', type: 'loading', content: '...' });

    // Build the trimmed history to send (last MAX_HISTORY_PAIRS pairs)
    const trimmedHistory = history.slice(-(MAX_HISTORY_PAIRS * 2));

    try {
      const response = await sendChatMessage({
        query,
        style,
        visual_type: visualMode,
        history: trimmedHistory,
      });

      // Remove loading indicator
      setMessages(prev => prev.filter(m => m.id !== loadingId));

      let botMessage: ChatMessage = { id: generateId(), sender: 'bot', type: 'text' };
      let assistantContent = '';

      if (response.type === 'diagram' && response.image) {
        botMessage = {
          ...botMessage,
          type: 'diagram',
          diagramData: {
            image: response.image,
            code: response.code ?? '',
            explanation: response.explanation ?? '',
          },
        };
        assistantContent = `[Mindmap diagram for: ${query}]`;
      } else if (response.type === 'video' && response.video_data) {
        botMessage = { ...botMessage, type: 'video', videoData: response.video_data };
        assistantContent = `[Video: ${response.video_data.title}]`;
      } else if (response.type === 'audio' && response.response) {
        // Fetch TTS audio
        let audioUrl: string | undefined;
        try {
          const ttsResult = await textToSpeech(response.response, sessionId);
          if (ttsResult.audio_url) {
            audioUrl = ttsResult.audio_url;
          } else if (ttsResult.blob) {
            audioUrl = URL.createObjectURL(ttsResult.blob);
          }
        } catch (_) { /* audio is optional */ }
        botMessage = { ...botMessage, type: 'audio', content: response.response, audioUrl };
        assistantContent = response.response;
      } else if (response.type === 'resource') {
        botMessage = {
          ...botMessage,
          type: 'resource',
          content: response.response,
          resourceData: response.resource,
        };
        assistantContent = response.response ?? '';
      } else {
        botMessage = { ...botMessage, type: 'text', content: response.response };
        assistantContent = response.response ?? '';
      }

      addMessage(botMessage);

      // Update conversation history for context preservation
      setHistory(prev => [
        ...prev,
        { role: 'user', content: query },
        { role: 'assistant', content: assistantContent },
      ]);
    } catch (err) {
      setMessages(prev => prev.filter(m => m.id !== loadingId));
      addMessage({
        id: generateId(),
        sender: 'bot',
        type: 'text',
        content: `❌ Error: ${String(err)}. Please try again.`,
      });
    } finally {
      setIsSending(false);
    }
  }, [history, style, visualMode, isSending, addMessage, sessionId]);

  return {
    messages,
    style,
    visualMode,
    isSending,
    sendMessage,
    changeStyle,
    setVisualMode,
  };
}
