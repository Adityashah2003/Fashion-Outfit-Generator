import React from 'react';

function ProductRecommendations({ recommendedProducts, handleThumbUp, handleRemoveProduct, dataReceived }) {
  return (
    <div className="product-recommendations">
      {dataReceived ? (
        <div className="product-card-container">
          {recommendedProducts.map((product, index) => (
            <div key={index} className="product-card">
              <img src={product.image} alt={product.name} />
              <span className='product-text'>{product.name}</span>
              <div>
                <button className='button-1' onClick={() => handleThumbUp(index)}>👍🏻</button>
                <button className='button-2' onClick={() => handleRemoveProduct(index)}>👎🏻</button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div></div>
      )}
    </div>
  );
}

export default ProductRecommendations;
