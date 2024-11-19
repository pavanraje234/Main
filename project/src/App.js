import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AccountAutomationApp from './components/pages/Account-Automation/App';
import FraudDetectionApp from './components/pages/Fraud-Detection/App';
import UploadApp from './components/pages/Main/App';
import DataView from './components/pages/Main/DataTable'
import MainApp from './components/App'
import Chatbot from './components/Chatbot';

function App() {
  return (
    <Router>
      <div>
        <Routes>
          <Route path="/account-automation" element={<AccountAutomationApp />} />
          <Route path="/fraud-detection" element={<FraudDetectionApp />} />
          <Route path="/data" element={<DataView/>}/>
          <Route path="/upload" element={<UploadApp />} />
          <Route path="/" element={<MainApp />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
