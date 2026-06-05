// api/assessmentApi.ts — Typed fetch wrappers for assessment routes

import type { AnswerRequest, QuestionResponse, SubmitAnswerResponse } from '../types';

const BASE = '/api';

export async function fetchNextQuestion(index: number): Promise<QuestionResponse> {
  const res = await fetch(`${BASE}/next_question?index=${index}`);
  if (!res.ok) throw new Error(`Failed to fetch question: ${res.status}`);
  return res.json();
}

export async function submitAnswer(body: AnswerRequest): Promise<SubmitAnswerResponse> {
  const res = await fetch(`${BASE}/submit_answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Failed to submit answer: ${res.status}`);
  return res.json();
}
