![mqdefault](https://github.com/user-attachments/assets/b9ae07e7-46f5-499d-9de0-43a61410fa45)


   after cloning it 
   open terminal
   cd baackend 
   python3 -m pip install fastapi uvicorn ultralytics torch torchvision torchaudio torch-geometric pydantic numpy opencv-python pillow

   python3 -m uvicorn main:app
   
   
   new teminal
   cd smart_traffic
   npm run dev

   you will be good to go 

   it have 5 fratures 

   # Smart Traffic Management System

## Overview
This project implements a **Smart Traffic Management System** that utilizes **AI-driven monitoring**, **dynamic traffic signal control**, and **time-series forecasting** to optimize urban traffic flow. It integrates **real-time computer vision** (YOLOv8) and **graph-based deep learning models** (ST-GCN, DCRNN) for traffic prediction and control.

---

## 1. Traffic Monitoring (Junction as a Graph Node)
Each **junction** is modeled as a **graph node**, where each lane is represented as an edge. **Each node** has **1 to 8 cameras** monitoring different lanes, and the system processes data **every 10 minutes**.

### 1.1 Real-time Traffic Analytics (YOLOv8)
Each camera runs **3 separate YOLOv8 models** for different tasks:

#### (1) Vehicle Detection, Speed Estimation, and Congestion Level
- **Model:** YOLOv8 Object Detection
- **Input:** Live camera feed  
- **Output:**  
  - **Vehicle count** per lane  
  - **Estimated speed** using object displacement (frame-to-frame tracking)  
  - **Congestion level**: Classified into `Low`, `Medium`, `High`, or `Extreme` using vehicle density heuristics  

#### (2) Helmet Violation Detection
- **Model:** YOLOv8 with a custom dataset trained to detect:  
  - **Helmet vs. No-Helmet classification**  
  - **Motorcycle rider’s position & face detection**  
- **Action:** Isolates images of violators and sends them to law enforcement.  

#### (3) Accident Detection
- **Model:** YOLOv8-segmentation (Instance Segmentation)
- **Logic:**  
  - Identifies abnormal vehicle positions (e.g., flipped cars, crashes)  
  - Detects **stationary vehicles** in live video  
  - Uses **object tracking** (ByteTrack/DeepSORT) to detect sudden halts  
- **Action:** Sends alerts to emergency services with real-time location.  

### 1.2 Aggregated Data Storage in a Time-Series Database
Every 10 minutes, the system aggregates data across all cameras and updates a **Time-Series Database (TSDB)** (e.g., **InfluxDB, TimescaleDB, or Apache Kafka** for real-time processing). The stored metrics include:
- **Total vehicle count per junction**  
- **Average congestion level (per lane & junction)**  
- **Helmet violations & accident reports**  
- **Average vehicle speed**  

This data is used for **forecasting & traffic signal optimization**.

---

## 2. Smart Traffic Light Control (12-Minute Cycle)
Each junction has **4 lanes (C1 to C4)**, and signal timing is dynamically adjusted based on traffic density.

### 2.1 Dynamic Traffic Light Scheduling Algorithm
The **total signal cycle** is **12 minutes (720 sec)**. Each lane receives:
- **Base time:** **90 seconds minimum**  
- **Dynamic allocation:** Remaining **360 seconds distributed** based on congestion level  

#### Step 1: Compute Traffic Density Per Lane
Example:
| Lane | Vehicles Count | Congestion Level |
|------|--------------|----------------|
| C1 | 40 | Low |
| C2 | 30 | Medium |
| C3 | 20 | High |
| C4 | 10 | Extreme |

Total Vehicles = **100**  
Each lane’s percentage of total traffic:  
- C1: **40%**  
- C2: **30%**  
- C3: **20%**  
- C4: **10%**  

#### Step 2: Allocate Traffic Signal Timing
- **Total cycle = 720 sec**  
- **Base time for all lanes = 360 sec**  
- **Remaining 360 sec distributed based on congestion %**  

| Lane | Base Time (sec) | Dynamic Time (sec) | Final Green Time (sec) |
|------|---------------|----------------|------------------|
| C1 | 90 | 144 | **234 sec** |
| C2 | 90 | 108 | **198 sec** |
| C3 | 90 | 72 | **162 sec** |
| C4 | 90 | 36 | **126 sec** |

🚦 **Final Signal Order (Highest congestion first):**  
**C4 → C3 → C2 → C1**

---

## 3. Forecasting & Traffic Optimization
To improve traffic control, we integrate **time-series forecasting** using **graph-based neural networks**.

### 3.1 Short-Term Forecasting (ST-GCN)
**Model:** **Spatial-Temporal Graph Convolutional Network (ST-GCN)**  
- **Goal:** Predict traffic conditions for the **next 15–30 minutes**.  
- **How it Works:**  
  - **Graph Representation:** Junctions are **nodes**, and roads connecting them are **edges**.  
  - **Spatial Component (GCN):** Learns relationships between neighboring junctions using adjacency matrices.  
  - **Temporal Component (RNN):** Captures sequential patterns (e.g., morning vs. evening congestion).  
  - **Data Source:** Time-series data from **YOLOv8 outputs & TSDB**.  
- **Use Case:** Adjusts signal timings dynamically based on upcoming congestion.

### 3.2 Long-Term Forecasting (DCRNN)
**Model:** **Diffusion Convolutional Recurrent Neural Network (DCRNN)**  
- **Goal:** Predict traffic conditions for the **next few hours/days** to assist city planners.  
- **How it Works:**  
  - **Graph Diffusion Convolution:** Models how traffic propagates across a city.  
  - **Recurrent Neural Network (RNN):** Captures long-term dependencies in traffic flow.  
  - **Data Source:** TSDB + historical traffic data.  
- **Use Case:**  
  - Predicts **road congestion trends**  
  - Helps city planners **optimize infrastructure changes**  

---

## 4. Emergency & Enforcement Handling
### 4.1 Helmet Violation Enforcement
- **Violators' images are sent to authorities.**  
- **License plate recognition (LPR)** can be integrated for automatic ticketing.

### 4.2 Automated Accident Detection & Response
- When an **accident is detected**, an **alert is automatically sent** to:  
  - **Emergency services**  
  - **Nearby junctions** (to adjust signals & reroute traffic)  

### 4.3 Intelligent Traffic Rerouting
- If **a junction remains at "Extreme" congestion for multiple cycles**, the system:  
  - **Automatically reroutes traffic** to alternative roads.  
  - **Informs nearby junctions** for better load balancing.  
  - **Adjusts signal durations dynamically** to prevent bottlenecks.  

---

## Why This Solution is Effective
✅ **Real-time monitoring & enforcement** (Helmet detection, accident alerts)  
✅ **Dynamic signal control** (Reduces congestion using live data)  
✅ **AI-based traffic forecasting** (Prevents future jams)  
✅ **Self-optimizing system** (Adapts to changing conditions)  

---

## Next Steps
- Implement a **Python prototype** for traffic monitoring & YOLOv8 integration.  
- Build the **traffic scheduling model** using **Dijkstra’s algorithm / Reinforcement Learning (RL)**.  
- Optimize **ST-GCN & DCRNN models** for forecasting accuracy. 🚦

