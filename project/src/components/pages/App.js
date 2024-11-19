import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AccountAutomationApp from './components/pages/Account-Automation/App';
import FraudDetectionApp from './components/pages/Fraud-Detection/App';
import MainApp from './components/pages/Main/App';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/account-automation" element={<AccountAutomationApp />} />
        <Route path="/fraud-detection" element={<FraudDetectionApp />} />
        <Route path="/main" element={<MainApp />} />
      </Routes>
    </Router>
  );
}

export default App;
