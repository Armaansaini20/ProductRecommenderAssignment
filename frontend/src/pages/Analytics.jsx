// frontend/src/pages/AnalyticsRoute.jsx
import React, { useEffect, useState } from "react";

export default function AnalyticsRoute({ apiBase = "http://127.0.0.1:8000" }) {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const res = await fetch(`${apiBase}/analytics/summary`);
        const data = await res.json();
        setSummary(data);
      } catch (err) {
        console.error(err);
        setError("Failed to load analytics");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [apiBase]);

  if (loading) return <div className="p-6">Loading analytics...</div>;
  if (error) return <div className="p-6 text-red-500">{error}</div>;
  if (!summary) return <div className="p-6">No analytics available.</div>;

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Dataset Analytics</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-lg bg-white shadow">
          <div className="text-sm text-gray-500">Total products</div>
          <div className="text-2xl font-bold">{summary.total_products ?? "—"}</div>
        </div>
        <div className="p-4 rounded-lg bg-white shadow">
          <div className="text-sm text-gray-500">Unique categories</div>
          <div className="text-2xl font-bold">{summary.unique_categories ?? "—"}</div>
        </div>
        <div className="p-4 rounded-lg bg-white shadow">
          <div className="text-sm text-gray-500">Products with images</div>
          <div className="text-2xl font-bold">{summary.products_with_images ?? "—"}</div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold mb-2">Top categories</h3>
        <table className="w-full text-sm">
          <thead className="text-gray-500">
            <tr>
              <th className="text-left py-1">Category</th>
              <th className="text-right py-1">Count</th>
            </tr>
          </thead>
          <tbody>
            {(summary.top_categories || []).map((c, i) => (
              <tr key={i} className={i % 2 === 0 ? "bg-gray-50" : ""}>
                <td className="py-1">{c[0]}</td>
                <td className="py-1 text-right">{c[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
