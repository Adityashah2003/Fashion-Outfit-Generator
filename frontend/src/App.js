import './App.css';
import Chatbot from './Chatbot.js';
import Cart from './Cart.js';
import React, { useState, useEffect } from 'react'; // Import useEffect
import ProductRecommendations from './ProductRecommendations'; // Import the ProductRecommendations component

function App() {
  const [cartProducts, setCartProducts] = useState([]);
  const [messages, setMessages] = useState([]);
  const [recommendedProducts, setRecommendedProducts] = useState([]); // Add state for recommendedProducts
  const [dataReceived, setDataReceived] = useState(false);
  
  const handleSystemResponse = (responseText) => {
    const newSystemMessage = { text: responseText, isUser: false };
    setMessages((prevMessages) => [...prevMessages, newSystemMessage]);
    fetchRecommendedProducts();
  };

  async function fetchRecommendedProducts() {
    try {
      const response = await fetch(' http://127.0.0.1:5000/get_data');
      const data = await response.json();
      setRecommendedProducts(data);
      console.log(data);
      setDataReceived(true);
    } catch (error) {
      console.error('Error fetching recommended products:', error);
    }
  }

  const handleThumbUp = (index) => {
    const productToMove = recommendedProducts[index];
    setCartProducts([...cartProducts, productToMove]);

    // Remove the product from recommendedProducts
    //updated
    const updatedRecommendedProducts = recommendedProducts.filter((_, i) => i !== index);
    setRecommendedProducts(updatedRecommendedProducts);
  };

  const handleRemoveProduct = (index) => {
    const updatedRecommendedProducts = recommendedProducts.filter((_, i) => i !== index);
    setRecommendedProducts(updatedRecommendedProducts);
  };

  return (
    <div className="app">
      <div className="left-half">
        <Chatbot handleSystemResponse={handleSystemResponse} messages={messages} />
      </div>
      <div className="right-half">
        <ProductRecommendations
          recommendedProducts={recommendedProducts}
          handleThumbUp={handleThumbUp}
          handleRemoveProduct={handleRemoveProduct}
          dataReceived={dataReceived}
        />
        <Cart cartProducts={cartProducts} />
      </div>
    </div>
  );
  };  
export default App;