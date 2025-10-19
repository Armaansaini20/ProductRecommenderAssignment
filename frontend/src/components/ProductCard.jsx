// frontend/src/components/ProductCard.jsx
import React, { useState, useRef } from "react";
import { PLACEHOLDER } from "../utils/placeholder";

/**
 * ProductCard with GenAI description generator + recommend action
 *
 * Props:
 *  - item: product item returned from backend (contains .id, .score, .meta)
 *  - onRecommend: function(productId) invoked when Recommend pressed
 *  - apiBase: optional base URL for API (default http://127.0.0.1:8000)
 */
export default function ProductCard({ item, onRecommend, apiBase = "http://127.0.0.1:8000" }) {
  const meta = item.meta || {};
  const img = meta.image_url || meta.images || meta.image || PLACEHOLDER;
  const imageSrc = Array.isArray(img) ? img[0] : img;

  // GenAI state
  const [genLoading, setGenLoading] = useState(false);
  const [genError, setGenError] = useState(null);
  const [generatedText, setGeneratedText] = useState(null);
  const [copied, setCopied] = useState(false);
  const copyTimeoutRef = useRef(null);

  // Recommend state (optional visual feedback)
  const [recLoading, setRecLoading] = useState(false);

  // Trigger recommend (calls parent callback)
  async function handleRecommend() {
    try {
      setRecLoading(true);
      await onRecommend && onRecommend(item.id);
    } finally {
      setRecLoading(false);
    }
  }

  // Call backend gen endpoint to create creative description
  async function handleGenerateDescription() {
    setGenError(null);
    setGeneratedText(null);
    setGenLoading(true);
    setCopied(false);
    try {
      // Compose payload: include title + short description + small meta to help generator
      const payload = {
  product_id: item.id,
  title: meta.title || "",
  description: meta.description || "",
  meta: { brand: meta.brand, material: meta.material, color: meta.color, categories: meta.categories },
  max_length: 120,
  style: "creative"
};
const res = await fetch(`${apiBase}/gen/description`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});


      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server returned ${res.status}: ${text}`);
      }

      const data = await res.json();
      // Accept both { description: "..." } or { generated: "..." }
      const desc = data.description ?? data.generated ?? data.output ?? null;
      if (!desc) throw new Error("No generated description returned");
      setGeneratedText(desc);
    } catch (err) {
      console.error("GenAI error:", err);
      setGenError(String(err.message || err));
    } finally {
      setGenLoading(false);
    }
  }

  function handleCopy() {
    if (!generatedText) return;
    navigator.clipboard.writeText(generatedText).then(() => {
      setCopied(true);
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = setTimeout(() => setCopied(false), 2500);
    });
  }

  // Cleanup copy timer on unmount
  React.useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    };
  }, []);

  return (
    <div className="bg-white rounded-2xl shadow p-4 hover:shadow-lg transition max-w-sm">
      <div className="w-full h-48 rounded-lg overflow-hidden bg-gray-100 flex items-center justify-center">
        <img
          src={imageSrc}
          alt={meta.title || item.id}
          className="object-contain w-full h-full"
          onError={(e) => { e.currentTarget.src = PLACEHOLDER; }}
        />
      </div>

      <div className="mt-3">
        <h3 className="text-sm font-semibold line-clamp-2">{meta.title || item.id}</h3>

        <div className="mt-2 flex items-center justify-between">
          <div className="text-xs text-gray-500">{meta.brand || meta.manufacturer || ""}</div>
          <div className="text-sm font-semibold">{meta.price ? `₹${meta.price}` : ""}</div>
        </div>

        <div className="mt-3 flex gap-2 items-center">
          <button
            onClick={handleRecommend}
            className={`text-xs px-3 py-1 rounded-full bg-gray-100 hover:bg-gray-200 transition ${recLoading ? "opacity-60" : ""}`}
            disabled={recLoading}
            title="Find similar products"
          >
            {recLoading ? "Searching..." : "Recommend"}
          </button>

          <button
            onClick={handleGenerateDescription}
            className={`text-xs px-3 py-1 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 transition ${genLoading ? "opacity-60" : ""}`}
            disabled={genLoading}
            title="Generate a creative product description"
          >
            {genLoading ? "Generating..." : "Generate Description"}
          </button>

          <div className="ml-auto text-xs text-gray-500">score: {item.score?.toFixed(3)}</div>
        </div>

        {/* generated description */}
        <div className="mt-4">
          {genLoading && (
            <div className="text-xs text-gray-600">✳️ Generating creative description...</div>
          )}

          {genError && (
            <div className="text-xs text-red-500">Error: {genError}</div>
          )}

          {generatedText && (
            <div className="mt-2 p-3 rounded-lg bg-gray-50 border border-gray-100">
              <div className="flex items-start justify-between gap-3">
                <div className="prose prose-sm max-w-none">
                  <p className="text-sm text-gray-800 whitespace-pre-line">{generatedText}</p>
                </div>
                <div className="flex flex-col items-end gap-2">
                  {/* <button
                    onClick={handleCopy}
                    className="text-xs px-2 py-1 rounded-full bg-gray-100 hover:bg-gray-200"
                  >
                    {copied ? "Copied" : "Copy"}
                  </button> */}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* short description fallback */}
        {!generatedText && meta.description && (
          <div className="mt-3 text-sm text-gray-600 line-clamp-3">{meta.description}</div>
        )}
      </div>
    </div>
  );
}
