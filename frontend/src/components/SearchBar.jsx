import React, { useState } from "react";

export default function SearchBar({ onSearch, placeholder="Search products..." }) {
  const [q, setQ] = useState("");

  const submit = (e) => {
    e?.preventDefault();
    if (!q.trim()) return;
    onSearch(q.trim());
  };

  return (
    <form onSubmit={submit} className="flex gap-3">
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder={placeholder}
        className="flex-1 px-4 py-3 rounded-2xl shadow-sm border border-gray-200 focus:outline-none focus:ring-2 focus:ring-brand-500"
      />
      <button
        type="submit"
        className="px-5 py-2 rounded-2xl bg-brand-500 text-black shadow hover:brightness-95"
      >
        Search
      </button>
    </form>
  );
}
