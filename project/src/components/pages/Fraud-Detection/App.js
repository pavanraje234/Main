import React, { useState } from 'react';
import './App.css';
import axios from 'axios';
import Chatbot from '../../Chatbot';
import { Link } from 'react-router-dom';
function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [dataRows, setDataRows] = useState([]);  // Initialize as an empty array
  const [error, setError] = useState('');
  const [metrics, setMetrics] = useState(null);
  
  // Handle file input
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  // Handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');

    if (!file) {
      setError("Please upload a file.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/fraud-detection', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log("Response from backend:", response.data);

      if (response.data) {
        setResult(response.data);
        setDataRows(response.data.preview || []);  // Set to empty array if undefined
        setMetrics(response.data.metrics);
        alert(`Fraud count detected: ${response.data.fraud_count}`);
      } else {
        console.error("No data returned from the server");
      }
    } catch (error) {
      console.error("There was an error with the request:", error.response?.data || error.message);
      setError("Error: " + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    window.location.href = 'https://demandion-88po.vercel.app/download';
  };

  return (
    <div className="App">
      <div className="sidebar" style={{float:"left", marginRight:"100px",marginLeft:"0px"}}>
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
  {/* Wrap the icons inside a Link component */}
  <Link to="/"> {/* Navigate to the main page */}
    <i className="fas fa-arrow-left"></i>
    <i className="fas fa-bolt"></i>
  </Link>
</div>
                <div className="">
                    <Chatbot />
                 
                </div>
            </div>
      <h1>Fraud Detection System</h1>
      <form onSubmit={handleSubmit}>
        <div className="upload-section">
          <label><strong>Upload Dataset (CSV):</strong></label><br />
          <input type="file" onChange={handleFileChange} />
          {error && <p style={{ color: 'red' }}>{error}</p>}
          <button type="submit" disabled={loading}>
            {loading ? "Processing..." : "Submit"}
          </button>
        </div>
      </form>

      {result && (
        <div className="result-section">
          <h2>Fraud Detection Results</h2>
          <p><strong>Fraud Count: </strong>{result.fraud_count}</p>

          {metrics && (
            <div className="metrics-section">
              <h3>Evaluation Metrics:</h3>
              <p><strong>Accuracy:</strong> {metrics.accuracy}</p>
              <p><strong>Precision:</strong> {metrics.precision}</p>
              <p><strong>Recall:</strong> {metrics.recall}</p>
              <p><strong>F1 Score:</strong> {metrics.f1_score}</p>
              <p><strong>ROC AUC Score:</strong> {metrics.roc_auc}</p>
            </div>
          )}

          <h3>Preview of the dataset (Fraud rows highlighted):</h3>
          {dataRows && dataRows.length > 0 && (
            <table>
              <thead>
                <tr>
                  {Object.keys(dataRows[0]).map((key) => (
                    <th key={key}>{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataRows.map((row, index) => {
                  // Highlight row based on fraud probability
                  let bgColor = 'black';  // Default color
                  if (row.predicted_fraud === 1) {
                    bgColor = 'red';  // Highlight if fraud value is 1
                  } else if (row.fraud_probability >= 0.5) {
                    bgColor = 'orange';  // Highlight if probability is greater than or equal to 0.5
                  }
                  return (
                    <tr key={index} style={{ backgroundColor: bgColor }}>
                      {Object.values(row).map((value, i) => (
                        <td key={i}>{value}</td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          <button onClick={handleDownload}>Download Result CSV</button>
        </div>
      )}
    </div>
  );
}

export default App;
