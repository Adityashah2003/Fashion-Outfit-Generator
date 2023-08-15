import React from 'react';

function ProductRecommendations({ recommendedProducts, handleThumbUp, handleRemoveProduct }) {
  return (
    <div className="product-recommendations">
      <div className="product-card-container">
        {recommendedProducts.map((product, index) => (
          <div key={index} className="product-card">
            <img src={`https://lp2.hm.com/hmgoepprod?set=format%5Bwebp%5D%2Cquality%5B79%5D%2Csource%5B%2Fc1%2Fe2%2Fc1e2fc11bd9fbbbc4a2534df25d06997abcb537d.jpg%5D%2Corigin%5Bdam%5D%2Ccategory%5Bmen_hoodiessweatshirts_hoodies%5D%2Ctype%5BDESCRIPTIVESTILLLIFE%5D%2Cres%5Bm%5D%2Chmver%5B2%5D&call=url%5Bfile%3A%2Fproduct%2Fmain%5D${product}`} alt={product} />
            <span>{product}</span>
            <button onClick={() => handleThumbUp(index)}>Thumbs Up</button>
            <button onClick={() => handleRemoveProduct(index)}>❌</button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ProductRecommendations;

