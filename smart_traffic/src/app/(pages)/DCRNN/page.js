"use client";
import { useState } from "react";

export default function TrafficPrediction() {
  const [hour, setHour] = useState(12);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);

  const fetchPredictions = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/dcrnn/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hour }),
      });
      const data = await response.json();
      setPredictions(data);
    } catch (error) {
      console.error("Error fetching predictions:", error);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: "20px", maxWidth: "500px", margin: "auto", backgroundColor: "white", borderRadius: "10px", boxShadow: "0px 0px 10px rgba(0,0,0,0.1)" }}>
      <h2 style={{ fontSize: "20px", fontWeight: "bold" }}>Traffic Prediction</h2>
      <div style={{ display: "flex", gap: "10px", marginTop: "10px" }}>
        <input
          type="number"
          value={hour}
          onChange={(e) => setHour(Number(e.target.value))}
          style={{ padding: "10px", border: "1px solid #ccc", borderRadius: "5px", flex: "1" }}
        />
        <button 
          onClick={fetchPredictions} 
          disabled={loading} 
          style={{ padding: "10px 15px", backgroundColor: "#007bff", color: "white", border: "none", borderRadius: "5px", cursor: "pointer" }}
        >
          {loading ? "Predicting..." : "Get Prediction"}
        </button>
      </div>
      {predictions && (
        <div style={{ marginTop: "15px" }}>
          <h3 style={{ fontWeight: "bold" }}>Predictions</h3>
          <pre style={{ padding: "10px", backgroundColor: "#f4f4f4", borderRadius: "5px", fontSize: "14px" }}>
            {JSON.stringify(predictions, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
