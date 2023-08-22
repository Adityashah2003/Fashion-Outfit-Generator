import React from 'react';

function Cart({ cartProducts }) {
  return (
    <div className="cart">
      <h2 className='cart-heading'>CART</h2>
      {cartProducts.map((product, index) => (
        <div key={index} className="cart-item">
          <span>{product.name}</span> {cartProducts.name}
          <img src={product.image} alt={product.name} /> {cartProducts.image}
          <a href={"https://www.flipkart.com"+product.links} target='_blank' class="buy-button">Buy Now</a>
        </div>
      ))}
    </div>
  );
}

export default Cart;
