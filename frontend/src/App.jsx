// src/App.jsx
import React from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Home from "./pages/Home";
import Analytics from "./pages/Analytics";

function Header() {
  return (
    <header className="w-full py-6 px-8 flex items-center justify-between border-b bg-white">
      <div className="flex items-center gap-4">
        <Link to="/" className="text-2xl font-bold text-slate-900">Ikarus3D — Product Recommender</Link>
        <nav className="hidden md:flex items-center gap-3 ml-6">
          <Link to="/" className="text-sm text-slate-600 hover:text-slate-900">Home</Link>
        </nav>
      </div>

      {/* Right-side controls */}
      <div className="flex items-center gap-3">
        <Link to="/analytics" className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-600 text-white hover:bg-indigo-700">
          Analytics
        </Link>
      </div>
    </header>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Header />
        <main className="p-6">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
