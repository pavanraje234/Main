import React, { useState } from 'react';
import './Chatbot.css';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  const handleSendMessage = (e) => {
    if (e.key === 'Enter' && input.trim() !== '') {
      const userMessage = { type: 'user', text: input };
      setMessages((prevMessages) => [...prevMessages, userMessage]);

      // Call the Flask API running on localhost:5000/chatbot
      fetch('http://localhost:5000/chatbot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: input }),
      })
        .then((response) => response.json())
        .then((data) => {
          const botMessage = { type: 'bot', text: data.answer };
          setMessages((prevMessages) => [...prevMessages, botMessage]);
        })
        .catch((error) => {
          console.error('Error:', error);
          const errorMessage = { type: 'bot', text: 'Error occurred while fetching response from bot' };
          setMessages((prevMessages) => [...prevMessages, errorMessage]);
        });

      setInput(''); // Clear input field
    }
  };

  return (
    <>
      <button id="chatbot-btn" onClick={toggleChatbot}>
        💬
      </button>

      {isOpen && (
        <div className="chatbot-container">
          <div className="chatbot-header">Chatbot</div>
          <div className="chatbot-messages" id="chatbot-messages">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`chatbot-message ${message.type === 'user' ? 'user-message' : 'bot-message'}`}
              >
                {message.type === 'user' ? 'You: ' : 'Bot: '}
                {message.text}
              </div>
            ))}
          </div>
          <input
            type="text"
            className="chatbot-input"
            placeholder="Type a message and press Enter..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleSendMessage}
          />
        </div>
      )}
    </>
  );
};

export default Chatbot;
