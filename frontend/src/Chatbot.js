import React, { useState, useEffect } from 'react';

function Chatbot({ handleSystemResponse }) {
  const [messages, setMessages] = useState([]);
  const [messageText, setMessageText] = useState('');

  const handleSendMessage = () => {
    if (messageText.trim() === '') return;

    const newUserMessage = { text: messageText, isUser: true };
    setMessages([...messages, newUserMessage]);

    setMessageText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSendMessage();
    }
  };

  useEffect(() => {
    if (messages.length > 0 && messages[messages.length - 1].isUser) {
      // Simulate system response after a delay
      setTimeout(() => {
        const newSystemMessage = { text: 'This is a system response.', isUser: false };
        setMessages([...messages, newSystemMessage]);
        handleSystemResponse(); // Add this line to trigger product recommendations
      }, 1000);
    }
  }, [messages]);

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
