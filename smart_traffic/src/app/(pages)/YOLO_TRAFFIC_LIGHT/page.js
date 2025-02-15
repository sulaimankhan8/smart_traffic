"use client";
import { useState } from "react";

const VEHICLE_WEIGHTS = {
  auto: 1,
  lcv: 1,
  motorcycle: 1,
  car: 2,
  tractor: 2,
  bus: 4,
  multiaxle: 4,
  truck: 4,
};

const MAX_CONGESTION = 20;

export default function Home() {
  const [images, setImages] = useState({ c1: null, c2: null, c3: null, c4: null });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [signalData, setSignalData] = useState(null);

  const handleImageChange = (event, field) => {
    const file = event.target.files[0];
    setImages((prev) => ({ ...prev, [field]: file }));
  };

  const handleUpload = async () => {
    if (!images.c1 || !images.c2 || !images.c3 || !images.c4) {
      alert("Please upload all 4 images.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    Object.keys(images).forEach((key) => formData.append("images", images[key]));

    try {
      const response = await fetch("http://localhost:8000/vehicle/detect/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process images.");
      }

      const data = await response.json();
      setResults(data);
      calculateTraffic(data);
    } catch (error) {
      console.error("Error:", error);
      alert("Error processing images.");
    }

    setLoading(false);
  };

  const calculateTraffic = (data) => {
    const laneCongestions = {};
    let totalVehicles = 0;

    Object.entries(data).forEach(([lane, details]) => {
      let congestion = 0;
      Object.entries(details.vehicles_detected).forEach(([type, count]) => {
        congestion += (VEHICLE_WEIGHTS[type] || 0) * count;
      });

      laneCongestions[lane] = congestion;
      totalVehicles += congestion;
    });

    const lanePercentages = {};
    Object.keys(laneCongestions).forEach((lane) => {
      lanePercentages[lane] = totalVehicles > 0 ? (laneCongestions[lane] / totalVehicles) * 100 : 0;
    });

    const totalCycleTime = 720;
    const baseTime = 90;
    const remainingTime = totalCycleTime - 4 * baseTime;

    const laneTimes = {};
    Object.keys(laneCongestions).forEach((lane) => {
      laneTimes[lane] = baseTime + (lanePercentages[lane] / 100) * remainingTime;
    });

    const sortedLanes = Object.entries(laneCongestions)
      .sort((a, b) => b[1] - a[1])
      .map(([lane]) => lane);

    setSignalData({ laneTimes, signalOrder: sortedLanes });
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-5">
      <h1 className="text-2xl font-bold mb-4">Vehicle Detection & Traffic Signal System</h1>

      <div className="grid grid-cols-2 gap-4">
        {["c1", "c2", "c3", "c4"].map((field, index) => (
          <div key={index} className="flex flex-col items-center">
            <input type="file" accept="image/*" onChange={(e) => handleImageChange(e, field)} />
            {images[field] && (
              <img
                src={URL.createObjectURL(images[field])}
                alt={`Preview ${index + 1}`}
                className="w-32 h-32 object-cover rounded-lg mt-2"
              />
            )}
          </div>
        ))}
      </div>

      <button onClick={handleUpload} disabled={loading} className="bg-blue-500 text-white px-4 py-2 rounded mt-4">
        {loading ? "Processing..." : "Upload & Detect"}
      </button>

      {results && (
        <div className="mt-6 p-4 bg-white shadow-md rounded">
          <h2 className="text-xl font-semibold mb-2">Detection Results:</h2>
          {Object.entries(results).map(([key, value]) => (
            <div key={key} className="mb-4 border p-4 rounded-lg shadow">
              <h3 className="font-bold text-lg mb-2">{key.toUpperCase()}</h3>
              <p>Total Vehicles: {value.total_vehicles}</p>
              <ul>
                {Object.entries(value.vehicles_detected).map(([type, count]) => (
                  <li key={type}>{type}: {count}</li>
                ))}
              </ul>
              {value.processed_image && (
                <img src={`data:image/jpeg;base64,${value.processed_image}`} alt={`Processed ${key}`} className="mt-4 w-64 h-auto rounded-lg shadow" />
              )}
            </div>
          ))}
        </div>
      )}

      {signalData && (
        <div className="mt-6 p-4 bg-green-200 shadow-md rounded">
          <h2 className="text-xl font-semibold mb-2">Traffic Signal Timing:</h2>
          <ul>
            {Object.entries(signalData.laneTimes).map(([lane, time]) => (
              <li key={lane}>{lane.toUpperCase()}: {Math.round(time)} sec</li>
            ))}
          </ul>
          <h3 className="mt-3 font-semibold">🚦 Signal Order:</h3>
          <p>{signalData.signalOrder.join(" → ")}</p>
        </div>
      )}
    </div>
  );
}
