import React from 'react';
import { Link } from 'react-router-dom';  // Use Link for SPA routing
import Chatbot from './Chatbot';
import './styles.css';

const MainApp = () => {
  return (
    <div className="container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="icon">
          <i className="fas fa-headset"></i>
        </div>
        <div className="icon">
          <i className="fas fa-check"></i>
        </div>
        <div className="icon">
          <i className="fas fa-users"></i>
        </div>
        <div className="icon">
          <a href="http://127.0.0.1:5000" target="_blank" rel="noopener noreferrer">
            <i className="fas fa-comment"></i>
          </a>
        </div>
        <div className="icon">
          <i className="fas fa-cog"></i>
        </div>
        <div className="icon">
          <i className="fas fa-arrow-left"></i>
          <i className="fas fa-bolt"></i>
          <Link to="/"></Link>
        </div>
        <div className="">
          <Chatbot />
        </div>
      </div>

      {/* Main Content */}
      <div className="main">
        <h1>Welcome to Demandion!</h1>
        <p>Choose the Service that you are looking for</p>
        <div className="options">
          <div className="option">
            <label htmlFor="file-upload">
              <i className="fas fa-upload"></i>
              {/* Use Link for routing */}
              <Link to="/upload">
                <p>Upload a Database</p>
              </Link>
            </label>
          </div>
          <div className="option">
            <i className="fas fa-cloud"></i>
            {/* Use Link for routing */}
            <Link to="/fraud-detection">
              <p>Fraud Detection</p>
            </Link>
          </div>
          <div className="option">
            <i className="fas fa-database"></i>
            {/* Use Link for routing */}
            <Link to="/account-automation">
              <p>Account Automation</p>
            </Link>
          </div>
          <div className="option">
            <i className="fas fa-file-upload"></i>
            <a href="https://demandion-88po.vercel.app/aivisualization" target="_blank" rel="noopener noreferrer">
              <p>AI Visualization</p>
            </a>
          </div>
          <div className="option">
            <i className="fas fa-link"></i>
            <a href="https://demandion-88po.vercel.app/" target="_blank" rel="noopener noreferrer">
              <p>Graph Generator</p>
            </a>
          </div>
          <div className="option">
            <i className="fas fa-sync-alt"></i>
            <a href="https://demandion-88po.vercel.app/recommendation" target="_blank" rel="noopener noreferrer">
              <p>Recommendations</p>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MainApp;
