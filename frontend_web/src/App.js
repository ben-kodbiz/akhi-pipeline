import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import TranscriptsList from './pages/TranscriptsList';
import TranscriptEditor from './pages/TranscriptEditor';
import JsonPreview from './pages/JsonPreview';
import QLoRA from './pages/QLoRA';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="transcripts" element={<TranscriptsList />} />
        <Route path="transcripts/:fileName" element={<TranscriptEditor />} />
        <Route path="json" element={<JsonPreview />} />
        <Route path="qlora" element={<QLoRA />} />
      </Route>
    </Routes>
  );
}

export default App;