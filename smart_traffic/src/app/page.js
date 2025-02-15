"use client"; // This makes sure everything in this file runs on the client

import dynamic from "next/dynamic";
import Navbar from "./components/Navbar";

// Dynamically import Map (client-side only)
const Map = dynamic(() => import("./components/Map"), { ssr: false });

const junctions = [
  { id: 2, name: "RVQV+X5W Lucknow", lat: 26.84, lon: 80.89294, congestion: "Low" },
  { id: 3, name: "Kanpur Rd, Alambagh", lat: 26.83117, lon: 80.91022, congestion: "Medium" },
  { id: 4, name: "RWMJ+9P6 Lucknow", lat: 26.83342, lon:80.93178, congestion: "High" },
  { id: 5, name: "Motilal Nehru Marg", lat: 26.83964, lon: 80.93472, congestion: "Extreme" },
  { id: 6, name: "Vidhan Sabha Marg, Husainganj", lat: 26.8429, lon: 80.9420 ,congestion: "Extreme"},
  { id: 7, name: "RWWW+JF8 Lucknow", lat: 26.8465, lon: 80.9462 ,congestion: "Extreme" },
  { id: 8, name: "VX42+5J7 Lucknow", lat: 26.8554, lon: 80.9516 ,congestion: "Extreme"},
];

export default function Home() {
  return (
    <div>
      <Navbar />
      <h1 className="text-3xl font-bold text-center my-4">Smart Traffic System</h1>
      <Map junctions={junctions}/>
      
      <section className="mt-8">
        <h2 className="text-xl font-semibold">Smart Traffic Management: Complete Solution</h2>
        <p className="text-lg">A smart traffic management system that combines real-time detection, short-term and long-term forecasting, and adaptive traffic signal control. Heres how it works:</p>
        
        <h3 className="text-lg font-semibold mt-6">1. Traffic Monitoring (Junction as a Node)</h3>
        <p>
          Each junction is a graph node with 1 to 8 cameras, each monitoring different lanes. Every 10 minutes, the system runs YOLOv8 models on each camera feed:
        </p>
        <ul className="list-disc pl-6">
          <li>Model 1: Vehicle count, congestion level, and speed estimation</li>
          <li>Model 2: Helmet detection (captures violators & alerts authorities)</li>
          <li>Model 3: Accident detection (isolates object & alerts emergency services)</li>
        </ul>

        <h3 className="text-lg font-semibold mt-6">2. Smart Traffic Light Control (12-Minute Cycle)</h3>
        <p>Each junction adjusts signal timing dynamically based on traffic conditions, ensuring optimal flow.</p>

        <h3 className="text-lg font-semibold mt-6">3. Forecasting & Optimization</h3>
        <p>
          The system updates a time-series database every 10 minutes for:
          <ul className="list-disc pl-6">
            <li>Short-Term Prediction (ST-GCNN): Predicts upcoming congestion in 15-30 minutes</li>
            <li>Long-Term Prediction (DCRNN): Optimizes infrastructure and predicts traffic patterns for hours/days</li>
          </ul>
        </p>

        <h3 className="text-lg font-semibold mt-6">4. Emergency & Enforcement Handling</h3>
        <p>Alerts are triggered in real-time for helmet violations, accidents, and high congestion, with the ability to reroute traffic when necessary.</p>

        <h3 className="text-lg font-semibold mt-6">Future Improvements</h3>
        <p className="text-lg">
          The system can be further enhanced by adopting advanced models and techniques:
        </p>
        <ul className="list-disc pl-6">
          <li>
            <strong>AST-GCN (Adaptive Spatio-Temporal Graph Convolutional Networks)</strong> – This model enables the system to learn spatial and temporal dependencies in traffic data more effectively, leading to better predictions and decision-making.
          </li>
          <li>
            <strong>Additional Behavioral Models</strong> – Incorporating models like <strong>Faster R-CNN</strong> for object detection and <strong>Assault detection</strong> for behavioral monitoring will improve accuracy in detecting traffic anomalies, accidents, or suspicious activities.
          </li>
          <li>
            <strong>City-Specific Database</strong> – A customized database specific to each city can enhance the accuracy of predictions. Localized data will allow better traffic pattern modeling and improve the systems adaptability to local conditions.
          </li>
          <li>
            <strong>Emergence of Foundation Models</strong> – As AI models evolve, foundation models trained on vast amounts of traffic and urban data could enhance the systems capability to predict long-term trends, plan infrastructure, and optimize city-wide traffic flow.
          </li>
        </ul>
        
        <h3 className="text-lg font-semibold mt-6">Why Your Solution is Effective</h3>
        <ul className="list-disc pl-6">
          <li>Real-time monitoring & enforcement (Helmet & accident detection)</li>
          <li>Dynamic traffic signal control (Reduces congestion)</li>
          <li>Predictive traffic management (Prevents future traffic jams)</li>
          <li>Adaptive system (Traffic signals adjust based on live data)</li>
        </ul>
      </section>
    </div>
  );
}
