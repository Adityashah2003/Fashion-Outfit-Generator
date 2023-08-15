import React, { useState } from 'react';
import './App.css';
import Chatbot from './Chatbot';
import ProductRecommendations from './ProductRecommendations';
import Cart from './Cart';

function App() {
  const [recommendedProducts, setRecommendedProducts] = useState([]);
  const [cartProducts, setCartProducts] = useState([]);

  const handleSystemResponse = () => {
    // Simulate recommended products from the system
    const newRecommendedProducts = ['Product A', 'Product B', 'Product C'];
    setRecommendedProducts([...recommendedProducts, ...newRecommendedProducts]);
  };

  const handleThumbUp = (index) => {
    const productToMove = recommendedProducts[index];
    setCartProducts([...cartProducts, productToMove]);

    // Remove the product from recommendedProducts
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
        <Chatbot handleSystemResponse={handleSystemResponse} />
      </div>
      <div className="right-half">
        <ProductRecommendations
          recommendedProducts={recommendedProducts}
          handleThumbUp={handleThumbUp}
          handleRemoveProduct={handleRemoveProduct}
        />
        <Cart cartProducts={cartProducts} />
      </div>
    </div>
  );
}

export default App;
