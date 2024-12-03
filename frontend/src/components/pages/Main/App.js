import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import { useNavigate } from 'react-router-dom';
import * as XLSX from 'xlsx';
import useDrivePicker from 'react-google-drive-picker'; // Added this for cloud storage
import './styles.css';
import Chatbot from '../../Chatbot';
import { Link } from 'react-router-dom';
const App = () => {
    const [fileData, setFileData] = useState(null);
    const [message, setMessage] = useState('');
    const [url, setUrl] = useState('');
    const [driveFiles, setDriveFiles] = useState([]); // Added this for cloud storage
    const [fileContent, setFileContent] = useState(''); // Added this for cloud storage
    const [openPicker] = useDrivePicker(); // Added this for cloud storage
    const navigate = useNavigate();

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append('file', file);

        try {
            let parsedData = [];
            if (file.type === 'text/csv') {
                Papa.parse(file, {
                    header: true,
                    complete: (results) => {
                        parsedData = results.data;
                        if (parsedData.length > 10) parsedData = parsedData.slice(0, 10);
                        navigate('/data', { state: { data: parsedData, type: 'csv' } });
                    },
                });
            } else if (file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
                const reader = new FileReader();
                reader.onload = (event) => {
                    const data = new Uint8Array(event.target.result);
                    const workbook = XLSX.read(data, { type: 'array' });
                    const sheet = workbook.Sheets[workbook.SheetNames[0]];
                    const sheetData = XLSX.utils.sheet_to_json(sheet, { header: 1 });
                    if (sheetData.length > 10) sheetData.splice(10);
                    navigate('/data', { state: { data: sheetData, type: 'excel' } });
                };
                reader.readAsArrayBuffer(file);
            } else {
                setMessage('Unsupported file type');
            }

            // Save file to server
            await axios.post('http://127.0.0.1:5000/uploadp', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
        } catch (error) {
            setMessage('File upload failed');
            console.error('Error uploading file:', error);
        }
    };

    const handleLinkSubmit = async () => {
        if (!url) {
            setMessage('Please enter a URL');
            return;
        }

        try {
            await axios.post('http://127.0.0.1:5000/upload-url', { url });
            setMessage('Data from URL has been saved successfully');

            // Fetch the data from the URL to display it
            const response = await axios.get(url);
            const data = response.data;

            const parsedData = Array.isArray(data) ? data.slice(0, 10) : [];
            navigate('/data', { state: { data: parsedData, type: 'link' } });
        } catch (error) {
            setMessage('Failed to fetch data from the URL');
            console.error('Error fetching data:', error);
        }
    };

    // Cloud storage logic added
    const handleOpenPicker = () => {
        openPicker({
            clientId: '172667069805-q2frm1q0ps9qeuleddjorjoc3obeamsq.apps.googleusercontent.com',
            developerKey: 'AIzaSyCw3vYrcCDG7mAAhVTa5j-62zcaH97suLU',
            viewId: 'DOCS',
            showUploadView: true,
            showUploadFolders: true,
            supportDrives: true,
            multiselect: true,
            callbackFunction: (data) => {
                if (data.action === 'picked') {
                    const selectedFiles = data.docs.map((file) => ({
                        name: file.name,
                        id: file.id,
                    }));
                    setDriveFiles(selectedFiles);

                    // Fetch content for the first selected file
                    const fileId = selectedFiles[0].id;
                    window.gapi.client.drive.files.get({
                        fileId: fileId,
                        alt: 'media',
                    }).then((response) => {
                        console.log('File content: ', response.body);
                        setFileContent(response.body);
                    }).catch((error) => {
                        console.error('Error fetching file content:', error);
                    });
                } else {
                    console.log('User closed the picker or cancelled');
                }
            },
        });
    };

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

            {/* Main Content */}
            <div className="main">
                <h1>Let's get started</h1>
                <p>Import your data in any of the following ways</p>
                <div className="options">
                    <div className="option">
                        <label htmlFor="file-upload">
                            <i className="fas fa-upload"></i>
                            <p>Upload from Computer</p>
                        </label>
                        <input
                            type="file"
                            id="file-upload"
                            style={{ display: 'none' }}
                            onChange={handleFileUpload}
                        />
                    </div>
                    <div className="option" onClick={handleOpenPicker}> {/* Cloud Storage button */}
                        <i className="fas fa-cloud"></i>
                        <p>Cloud Storage</p>
                    </div>
                    <div className="option">
                        <i className="fas fa-database"></i>
                        <p>Database</p>
                    </div>
                    <div className="option">
                        <i className="fas fa-file-upload"></i>
                        <p>Import from File</p>
                    </div>
                    <div className="option">
                        <i className="fas fa-link"></i>
                        <p>Import via Link</p>
                        <input
                            type="text"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="Enter data URL"
                        />
                        <button onClick={handleLinkSubmit}>Submit</button>
                    </div>
                    <div className="option">
                        <i className="fas fa-sync-alt"></i>
                        <p>Sync with Service</p>
                    </div>
                </div>

                {fileData && (
                    <div className="uploaded-data">
                        <h3>File Uploaded:</h3>
                        <p>{fileData}</p>
                    </div>
                )}

                {message && (
                    <div className="upload-message">
                        <p>{message}</p>
                    </div>
                )}

                {/* Display Google Drive content */}
                {driveFiles.length > 0 && (
                    <div>
                        <h3>Files from Google Drive:</h3>
                        <ul>
                            {driveFiles.map((file, index) => (
                                <li key={index}>{file.name}</li>
                            ))}
                        </ul>
                    </div>
                )}

                {fileContent && (
                    <div>
                        <h3>File Content from Google Drive:</h3>
                        <pre>{fileContent}</pre>
                    </div>
                )}
            </div>
        </div>
    );
};

export default App;
