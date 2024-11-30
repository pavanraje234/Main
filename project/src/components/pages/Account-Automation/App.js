import React, { useState } from 'react';
import Chatbot from '../../Chatbot';
import { Link } from 'react-router-dom';
const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [graphs, setGraphs] = useState([]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);
  
    const response = await fetch('https://demandion-88po.vercel.app/upload', {
        method: 'POST',
        body: formData,
      });
  
    if (response.ok) {
      const graphUrls = await response.json(); // Returns URLs for the graphs
      console.log(graphUrls); // Debugging here
      setGraphs(graphUrls);
    } else {
      console.error("File upload failed");
    }
  };
  
  return (
    
    <div style={{display:"flex"}}>
      {/* Sidebar */}
      <div className="sidebar" style={{float:"left", marginRight:"100px"}}>
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
                    <a href="https://demandion-88po.vercel.app/" target="_blank" rel="noopener noreferrer">
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
            <div style={{display:"flex", justifyContent:"center",alignItems:"center",flexDirection:"column",marginTop:"90px",marginLeft:"50px",font:"90px"}}>
      <h2 >Upload a CSV File</h2>
      <input type="file" onChange={handleFileChange} style={{marginLeft:"30px"}} />
      <button onClick={handleUpload} style={{margin:"30px"}}>Upload and Generate Graphs</button>
      <div >
    </div>
      </div>
      <div style={{marginRight:"00px",marginTop:"0px"}}>
        {graphs.map((url, index) => (
          <div key={index} style={{marginLeft:"0px",marginTop:"0px"}}>
            <h3 style={{ marginRight:"0px",marginTop:"0px"}}>Graph {index + 1}</h3>
            <img src={`https://demandion-88po.vercel.app${url}`} alt={`Graph ${index + 1}`} style={{marginTop:"50px"}} />
          </div>
        ))}
      </div>
    </div>
  );
};

export default FileUpload;
