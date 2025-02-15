"use client"; // If using App Router (Next.js 13+)

import { useState } from "react";
import axios from "axios";

export default function Home() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    // Handle image upload
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setImage(file);
            setPreview(URL.createObjectURL(file)); // Show image preview
        }
    };

    // Send image to backend
    const handleUpload = async () => {
        if (!image) return alert("Please select an image!");

        const formData = new FormData();
        formData.append("image", image);

        setLoading(true);
        try {
            const response = await axios.post("http://127.0.0.1:8000/helmet/helmet/detect/", formData, {
                headers: { "Content-Type": "multipart/form-data" }
            });

            setResult(response.data);
        } catch (error) {
            console.error("Error detecting helmet:", error);
        }
        setLoading(false);
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-screen p-4 bg-gray-100">
            <h1 className="text-2xl font-bold mb-4">Helmet Detection</h1>

            {/* Image Upload */}
            <input type="file" accept="image/*" onChange={handleFileChange} className="mb-4" />

            {/* Image Preview */}
            {preview && (
                <div className="relative">
                    <img src={preview} alt="Uploaded" className="w-80 h-auto rounded shadow" />
                </div>
            )}

            {/* Upload Button */}
            <button
                onClick={handleUpload}
                disabled={loading}
                className="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
                {loading ? "Processing..." : "Detect Helmet"}
            </button>

            {/* Display Results */}
            {result && (
                <div className="mt-4 bg-white p-4 rounded shadow">
                    <h2 className="text-lg font-semibold">Detection Results:</h2>
                    <p>Helmet Count: {result.helmet_count}</p>
                    <p>No Helmet Count: {result.no_helmet_count}</p>
                </div>
            )}
        </div>
    );
}
