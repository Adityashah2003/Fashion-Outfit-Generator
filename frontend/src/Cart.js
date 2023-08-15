import React from 'react';

function Cart({ cartProducts }) {
  return (
    <div className="cart">
      <h2>Cart</h2>
      {cartProducts.map((product, index) => (
        <div key={index} className="cart-item">
          <span>{product}</span>
          <button>Buy Now</button>
        </div>
      ))}
    </div>
  );
}

export default Cart;
