// pages/AssessmentPage.tsx — VARK quiz page

import { useNavigate } from 'react-router-dom';
import { useAssessment } from '../hooks/useAssessment';
import '../styles/assessment.css';

const ANSWERS = [
  'Strongly Agree',
  'Agree',
  'Neutral',
  'Disagree',
  'Strongly Disagree',
] as const;

type AnswerValue = typeof ANSWERS[number];

export default function AssessmentPage() {
  const navigate = useNavigate();
  const { question, questionIndex, totalQuestions, isLoading, error, result, answerQuestion } =
    useAssessment();

  if (result) {
    navigate(`/result?style=${encodeURIComponent(result.style)}&confidence=${result.confidence}`);
    return null;
  }

  return (
    <main className="assessment-page">
      <div className="blob-bg blob-1" />
      <div className="blob-bg blob-2" />
      <div className="assessment-card">
        {/* Header */}
        <div className="assessment-logo">
          <h1 className="gradient-text">Cognitive Chatbot</h1>
          <p>Discover your personalised ML learning style</p>
        </div>

        {/* Progress */}
        <div>
          <div className="progress-label">
            <span>Question {isLoading ? '…' : questionIndex} of {totalQuestions}</span>
            <span>{isLoading ? 0 : Math.round((questionIndex / totalQuestions) * 100)}%</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${(questionIndex / totalQuestions) * 100}%` }}
            />
          </div>
        </div>

        {/* Question */}
        {isLoading ? (
          <>
            <div className="skeleton-text" style={{ width: '85%' }} />
            <div className="skeleton-text" style={{ width: '65%', marginBottom: 36 }} />
          </>
        ) : error ? (
          <div className="error-msg">{error}</div>
        ) : (
          <p className="question-text animate-fade-in">{question}</p>
        )}

        {/* Answer buttons */}
        <div className="answer-grid">
          {ANSWERS.map((answer) => (
            <button
              key={answer}
              className="answer-btn"
              disabled={isLoading}
              onClick={() => answerQuestion(answer as AnswerValue)}
              id={`answer-${answer.toLowerCase().replace(/\s/g, '-')}`}
            >
              <span className="dot" />
              {answer}
            </button>
          ))}
        </div>
      </div>
    </main>
  );
}
