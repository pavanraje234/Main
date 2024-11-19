import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './styles.css';

const DataTable = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { data, type } = location.state || { data: [], type: '' };

    return (
        <div className="container">
            {/* Sidebar */}
            <div className="sidebar">
                <div className="icon"><i className="fas fa-headset"></i></div>
                <div className="icon"><i className="fas fa-check"></i></div>
                <div className="icon"><i className="fas fa-users"></i></div>
                <div className="icon"><i className="fas fa-comment"></i></div>
                <div className="icon"><i className="fas fa-cog"></i></div>
                <div className="icon"><i className="fas fa-arrow-left"></i><i className="fas fa-bolt"></i></div>
            </div>

            {/* Main Content */}
            <div className="main">
                <h1>Data Preview</h1>
                {data && data.length > 0 && (
                    <table>
                        <thead>
                            <tr>
                                {/* CSV and Cloud Data */}
                                {(type === 'csv' || type === 'cloud') && Object.keys(data[0]).map((key) => (
                                    <th key={key}>{key}</th>
                                ))}
                                {/* Excel Data */}
                                {type === 'excel' && data[0].map((cell, index) => (
                                    <th key={index}>{cell}</th>
                                ))}
                                {/* Link Data */}
                                {type === 'link' && Array.isArray(data[0]) && data[0].map((cell, index) => (
                                    <th key={index}>{cell}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {data.slice(1).map((row, index) => (
                                <tr key={index}>
                                    {/* CSV and Cloud Data */}
                                    {(type === 'csv' || type === 'cloud') && Object.values(row).map((cell, i) => (
                                        <td key={i}>{cell}</td>
                                    ))}
                                    {/* Excel Data */}
                                    {type === 'excel' && row.map((cell, i) => (
                                        <td key={i}>{cell}</td>
                                    ))}
                                    {/* Link Data */}
                                    {type === 'link' && row.map((cell, i) => (
                                        <td key={i}>{cell}</td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
                <button onClick={() => navigate('/')}>Back to Upload Page</button>
            </div>
        </div>
    );
};

export default DataTable;
