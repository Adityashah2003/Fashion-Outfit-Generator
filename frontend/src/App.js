import './App.css';
import Chatbot from './Chatbot';
import Cart from './Cart';
import React, { useState } from 'react';

function App() {
  const [cartProducts, setCartProducts] = useState([]);
  const [messages, setMessages] = useState([]); // Add this state for managing messages

  const handleSystemResponse = (responseText) => {
    const newSystemMessage = { text: responseText, isUser: false };
    setMessages((prevMessages) => [...prevMessages, newSystemMessage]);
  };

  // useEffect(() => {
  //   async function fetchRecommendedProducts() {
  //     try {
  //       const response = await fetch('/get_product_info');
  //       const data = await response.json();
  //       setRecommendedProducts(data);
  //     } catch (error) {
  //       console.error('Error fetching recommended products:', error);
  //     }
  //   }

  //   fetchRecommendedProducts();
  // }, []); // Empty dependency array means this effect runs once, on component mount

  // const handleThumbUp = (index) => {
  //   const productToMove = recommendedProducts[index];
  //   setCartProducts([...cartProducts, productToMove]);

  //   // Remove the product from recommendedProducts
  //   const updatedRecommendedProducts = recommendedProducts.filter((_, i) => i !== index);
  //   setRecommendedProducts(updatedRecommendedProducts);
  // };

  // const handleRemoveProduct = (index) => {
  //   const updatedRecommendedProducts = recommendedProducts.filter((_, i) => i !== index);
  //   setRecommendedProducts(updatedRecommendedProducts);
  // };

  return (
    <div className="app">
      <div className="left-half">
      <Chatbot handleSystemResponse={handleSystemResponse} messages={messages} />
      </div>
      <div className="right-half">
        <Cart cartProducts={cartProducts} />
      </div>
    </div>
  );
}

export default App;