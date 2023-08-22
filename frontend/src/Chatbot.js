import React, { useState, useEffect } from 'react';

function Chatbot({ handleSystemResponse }) {
  const [messages, setMessages] = useState([]);
  const [messageText, setMessageText] = useState('');

  const sendRequestToServer = async (serverPort) => {
    try {
      const response = await fetch(`http://127.0.0.1:${serverPort}/recvprompt`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: messageText }),
      });

      const responseData = await response.json();
      const systemResponse = responseData.response;

      // Call the function to handle the system response
      handleSystemResponse(systemResponse);

      const newSystemMessage = { text: systemResponse, isUser: false };
      setMessages((prevMessages) => [...prevMessages, newSystemMessage]);
    } catch (error) {
      console.error('Error sending message to backend:', error);
    }
  };

  const handleSendMessage = async () => {
    if (messageText.trim() === '') return;

    const newUserMessage = { text: messageText, isUser: true };
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);

    setMessageText('');

    await sendRequestToServer(5000); 
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div>
      <div className="chatbot">
        <div className="chat-messages">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.isUser ? 'user' : 'system'}`}>
              {message.text}
            </div>
          ))}
        </div>
        
      </div>
      <input
          className="sticky"
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
