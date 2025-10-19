import React from "react";
import ProductCard from "./ProductCard";

export default function ProductGrid({ items = [], onRecommend }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
      {items.map((it) => (
        <ProductCard key={it.id} item={it} onRecommend={onRecommend} />
      ))}
    </div>
  );
}
