"use client";
// app/components/Map.jsx
import { MapContainer, TileLayer, Circle, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';



const congestionColors = {
    Low: "green",
    Medium: "yellow",
    High: "orange",
    Extreme: "red",
  };
  
  const Map = ({ junctions }) => {
    return (
      <MapContainer center={[26.8400, 80.8929]} zoom={13} className="h-screen w-full">
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {junctions.map((junc) => (
          <Circle
            key={junc.id}
            center={[junc.lat, junc.lon]}
            radius={200} // Adjust radius for better visibility
            color={congestionColors[junc.congestion] || "gray"}
            fillColor={congestionColors[junc.congestion] || "gray"}
            fillOpacity={0.5}
          >
            <Popup>
              <strong>{junc.name}</strong><br />
              Congestion: {junc.congestion}
            </Popup>
          </Circle>
        ))}
        <Polyline positions={junctions.map(j => [j.lat, j.lon])} color="blue" />
      </MapContainer>
    );
  };
  
  export default Map;