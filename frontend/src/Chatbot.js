import React, { useState, useEffect } from 'react';

function Chatbot({ handleSystemResponse }) {
  const [messages, setMessages] = useState([]);
  const [messageText, setMessageText] = useState('');

  const handleSendMessage = async () => {
    if (messageText.trim() === '') return;
  
    const newUserMessage = { text: messageText, isUser: true };
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);
  
    setMessageText('');
  
    try {
      const response = await fetch('http://127.0.0.1:5000/recvprompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: messageText }),
      });
  
      const responseData = await response.json();
      const systemResponse = responseData.response; // Get the response from the server
  
      // Display the response as a system message
      const newSystemMessage = { text: systemResponse, isUser: false };
      setMessages((prevMessages) => [...prevMessages, newSystemMessage]);
    } catch (error) {
      console.error('Error sending message to backend:', error);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chatbot">
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.isUser ? 'user' : 'system'}`}>
            {message.text}
      </div>
      ))}
    </div>
      <input
        type="text"
        placeholder="Type your message..."
        value={messageText}
        onChange={(e) => setMessageText(e.target.value)}
        onKeyDown={handleKeyDown}
      />
    </div>
  );
} 

export default Chatbot;
