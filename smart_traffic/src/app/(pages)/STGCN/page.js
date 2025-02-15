"use client";
import { useState } from "react";



export default function STGCNPage() {
    const [inputData, setInputData] = useState([
        [50, 40, 0.5], 
        [10, 35, 0.1], 
        [90, 30, 0.9],   
        [20, 45, 0.2], 
        [80, 38, 0.8], 
        [100, 25, 1.0], 
        [55, 50, 0.55]
    ]);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);

    const fetchPrediction = async () => {
        setLoading(true);
        try {
            const response = await fetch("http://127.0.0.1:8000/stgcn/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ data: inputData }),
            });
            const result = await response.json();
            setPrediction(result);
        } catch (error) {
            console.error("Error fetching prediction:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-6 max-w-3xl mx-auto">
            <h1 className="text-2xl font-bold mb-4">STGCN Traffic Prediction</h1>
            
            <button 
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
                onClick={fetchPrediction}
                disabled={loading}
            >
                {loading ? "Predicting..." : "Get Prediction"}
            </button>

            {prediction && (
                <div className="mt-6 p-4 border rounded bg-gray-50">
                    <h2 className="text-lg font-semibold mb-2">Prediction Results:</h2>

                    {/* 10-Minute Predictions */}
                    {prediction["10_min"] && (
                        <div className="mt-4">
                            <h3 className="text-md font-bold mb-2">10-Minute Forecast</h3>
                            <table className="w-full border-collapse border border-gray-300">
                                <thead className="bg-gray-200">
                                    <tr>
                                        <th className="border border-gray-300 px-4 py-2">No. of Vehicles</th>
                                        <th className="border border-gray-300 px-4 py-2">Avg Speed (km/h)</th>
                                        <th className="border border-gray-300 px-4 py-2">Congestion Level</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {prediction["10_min"].map((row, index) => (
                                        <tr key={index} className="text-center">
                                            <td className="border border-gray-300 px-4 py-2">{row[0]}</td>
                                            <td className="border border-gray-300 px-4 py-2">{row[1]} km/h</td>
                                            <td className="border border-gray-300 px-4 py-2">{(row[2] * 100).toFixed(2)}%</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    {/* 20-Minute Predictions */}
                    {prediction["20_min"] && (
                        <div className="mt-6">
                            <h3 className="text-md font-bold mb-2">20-Minute Forecast</h3>
                            <table className="w-full border-collapse border border-gray-300">
                                <thead className="bg-gray-200">
                                    <tr>
                                        <th className="border border-gray-300 px-4 py-2">No. of Vehicles</th>
                                        <th className="border border-gray-300 px-4 py-2">Avg Speed (km/h)</th>
                                        <th className="border border-gray-300 px-4 py-2">Congestion Level</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {prediction["20_min"].map((row, index) => (
                                        <tr key={index} className="text-center">
                                            <td className="border border-gray-300 px-4 py-2">{row[0]}</td>
                                            <td className="border border-gray-300 px-4 py-2">{row[1]} km/h</td>
                                            <td className="border border-gray-300 px-4 py-2">{(row[2] * 100).toFixed(2)}%</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
