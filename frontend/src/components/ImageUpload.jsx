import React, { useRef, useState } from "react";

export default function ImageUpload({ onUpload }) {
  const inputRef = useRef();
  const [loading, setLoading] = useState(false);

  const handle = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setLoading(true);
    try {
      await onUpload(f);
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      setLoading(false);
      inputRef.current.value = null;
    }
  };

  return (
    <div className="flex items-center gap-3">
      <label className="cursor-pointer inline-flex items-center gap-2 px-4 py-2 rounded-2xl border border-dashed border-gray-200 hover:bg-gray-50">
        <input ref={inputRef} type="file" accept="image/*" onChange={handle} className="hidden" />
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V7M16 3l-4 4M12 3l4 4" /></svg>
        <span className="text-sm text-gray-700">{loading ? "Uploading..." : "Upload image"}</span>
      </label>
    </div>
  );
}
