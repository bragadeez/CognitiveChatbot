// App.tsx — Root router

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AssessmentPage from './pages/AssessmentPage';
import ResultPage     from './pages/ResultPage';
import ChatPage       from './pages/ChatPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"       element={<AssessmentPage />} />
        <Route path="/result" element={<ResultPage />} />
        <Route path="/chat"   element={<ChatPage />} />
        <Route path="*"       element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
