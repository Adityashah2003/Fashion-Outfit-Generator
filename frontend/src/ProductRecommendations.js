import React from 'react';

function ProductRecommendations({ recommendedProducts, handleThumbUp, handleRemoveProduct }) {
  return (
    <div className="product-recommendations">
      <div className="product-card-container">
        {recommendedProducts.map((product, index) => (
          <div key={index} className="product-card">
            <h4>{product.name}</h4>
            <img src={product.image} alt={product.name} />
            <span>{product.name}</span>
            <button onClick={() => handleThumbUp(index)}>Thumbs Up</button>
            <button onClick={() => handleRemoveProduct(index)}>❌</button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ProductRecommendations;
