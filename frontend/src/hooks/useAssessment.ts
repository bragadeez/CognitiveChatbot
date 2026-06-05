// hooks/useAssessment.ts — State machine for the VARK quiz flow
// Replaces the fragile server-side global state with typed React state.

import { useState, useCallback, useEffect } from 'react';
import { fetchNextQuestion, submitAnswer } from '../api/assessmentApi';
import type { AssessmentResult, LearningStyle } from '../types';

type AnswerValue = 'Strongly Agree' | 'Agree' | 'Neutral' | 'Disagree' | 'Strongly Disagree';

interface AssessmentState {
  question: string;
  questionIndex: number;
  totalQuestions: number;
  isLoading: boolean;
  error: string | null;
  result: AssessmentResult | null;
  sessionId: string;
}

// Generate or retrieve a persistent session UUID for this browser tab
function getSessionId(): string {
  let id = sessionStorage.getItem('cog_session_id');
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem('cog_session_id', id);
  }
  return id;
}

export function useAssessment() {
  const [state, setState] = useState<AssessmentState>({
    question: '',
    questionIndex: 0,
    totalQuestions: 15,
    isLoading: true,
    error: null,
    result: null,
    sessionId: getSessionId(),
  });

  const loadQuestion = useCallback(async (index: number) => {
    setState(s => ({ ...s, isLoading: true, error: null }));
    try {
      const data = await fetchNextQuestion(index);
      if (data.done && data.style) {
        setState(s => ({
          ...s,
          isLoading: false,
          result: { style: data.style as LearningStyle, confidence: data.confidence ?? 0 },
        }));
      } else {
        setState(s => ({
          ...s,
          isLoading: false,
          question: data.question ?? '',
          questionIndex: data.index ?? index,
          totalQuestions: data.total ?? 15,
        }));
      }
    } catch (err) {
      setState(s => ({ ...s, isLoading: false, error: String(err) }));
    }
  }, []);

  // Load first question on mount
  useEffect(() => { loadQuestion(0); }, [loadQuestion]);

  const answerQuestion = useCallback(async (answer: AnswerValue) => {
    setState(s => ({ ...s, isLoading: true, error: null }));
    try {
      const data = await submitAnswer({ answer, session_id: state.sessionId });
      if (data.done && data.style) {
        setState(s => ({
          ...s,
          isLoading: false,
          result: { style: data.style as LearningStyle, confidence: data.confidence ?? 0 },
        }));
      } else {
        await loadQuestion(state.questionIndex);
      }
    } catch (err) {
      setState(s => ({ ...s, isLoading: false, error: String(err) }));
    }
  }, [state.sessionId, state.questionIndex, loadQuestion]);

  return { ...state, answerQuestion };
}
