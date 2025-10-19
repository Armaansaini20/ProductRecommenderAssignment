import React, { useState } from "react";
import SearchBar from "../components/SearchBar";
import ImageUpload from "../components/ImageUpload";
import ProductGrid from "../components/ProductGrid";
import { postJSON, postImage, getJSON } from "../api/apiClient";

export default function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  async function searchText(q) {
    setLoading(true);
    try {
      const data = await postJSON("/recommend", { prompt: q, top_k: 12 });
      setResults(data.results || []);
    } catch (err) {
      alert("Search error: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  async function uploadImageFile(file) {
    setLoading(true);
    try {
      const data = await postImage("/search/image?top_k=12", file);
      setResults(data.results || []);
    } catch (err) {
      alert("Image search failed: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  async function recommendByProduct(productId) {
    setLoading(true);
    try {
      const data = await getJSON(`/recommend/${productId}?top_k=12`);
      setResults(data.results || []);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      alert("Recommend by product failed: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen p-1">
      <div className="max-w-6xl mx-auto">
        
        <section className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
          <div className="md:col-span-2">
            <SearchBar onSearch={searchText} />
            <div className="mt-3 text-xs text-gray-500">Try: "wooden chair", "bathroom hardware", etc.</div>
          </div>
          <div className="md:col-span-1 flex flex-col gap-3">
            <ImageUpload onUpload={uploadImageFile} />
            <button
              onClick={() => { setResults([]); setLoading(false); }}
              className="px-3 py-2 rounded-2xl bg-white border text-sm"
            >
              Clear
            </button>
          </div>
        </section>

        <section>
          <div className="mb-4 flex items-center justify-between">
            <div className="text-sm text-gray-600">{loading ? "Searching..." : `${results.length} results`}</div>
          </div>

          <ProductGrid items={results} onRecommend={recommendByProduct} />
        </section>
      </div>
    </div>
  );
}
