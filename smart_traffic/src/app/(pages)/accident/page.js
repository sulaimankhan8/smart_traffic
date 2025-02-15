"use client";
import { useState } from "react";

export default function Home() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [processedImage, setProcessedImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [imageName, setImageName] = useState("");

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            setImageName(file.name.replace(/\.[^/.]+$/, "") + "_processed.png");
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            alert("Please select an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);

        setLoading(true);
        try {
            const response = await fetch("http://127.0.0.1:8000/accident/upload/", {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const blob = await response.blob();
                setProcessedImage(URL.createObjectURL(blob));
            } else {
                alert("Error processing image.");
            }
        } catch (error) {
            alert("An error occurred while uploading the image.");
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const handleDownload = () => {
        if (processedImage) {
            const link = document.createElement("a");
            link.href = processedImage;
            link.download = imageName || "processed_image.png";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    return (
        <div style={{ textAlign: "center", padding: "20px" }}>
            <h1>Accident Detection</h1>
            <input type="file" accept="image/*" onChange={handleFileChange} />
            <button onClick={handleUpload} style={{ marginLeft: "10px" }} disabled={loading}>
                {loading ? "Processing..." : "Upload & Detect"}
            </button>

            {processedImage && (
                <div>
                    <h2>Processed Image:</h2>
                    <img src={processedImage} alt="Detected Image" style={{ maxWidth: "100%", border: "2px solid black" }} />
                    <br />
                    <button onClick={handleDownload} style={{ marginTop: "10px" }}>Download Processed Image</button>
                </div>
            )}
        </div>
    );
}
