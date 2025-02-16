![mqdefault](https://github.com/user-attachments/assets/b9ae07e7-46f5-499d-9de0-43a61410fa45)
https://www.youtube.com/watch?v=EVyDJPDPGGE

   after cloning it ,
   open terminal,
   cd baackend ,
   python3 -m pip install fastapi uvicorn ultralytics torch torchvision torchaudio torch-geometric pydantic numpy opencv-python pillow

   python3 -m uvicorn main:app
   
   
   new teminal
   cd smart_traffic
   npm run dev

   you will be good to go 

   

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

 # Smart Traffic Management System - Lucknow Implementation

## Overview
This document details the implementation of a **Smart Traffic Management System** for **Lucknow**, covering **395 junctions/intersections** categorized by traffic density:

- **High-Traffic Junctions:** 50
- **Mid-Traffic Junctions:** 345

## System Components

### Cameras
- **High-Traffic Areas:** Axis P1468-LE / Hikvision ANPR (₹45,000 per unit)
- **Mid-Traffic Areas:** Hikvision DS-2 CD2085G1-I (₹12,000 per unit)
- **Storage Requirement:** 20GB per camera per day

### On-Site Edge Processing
- **Jetson Xavier NX (₹45,000 per unit) for real-time AI processing**
- **Arduino for auxiliary control functions**

### AI Processing & Traffic Analysis
Each **Jetson Xavier NX** processes real-time traffic data every **10 minutes**, sending updates to a centralized database. The following parameters are captured:
1. **Vehicle Count** (Total vehicles detected in the last 10 minutes)
2. **Average Speed** (Calculated based on object tracking data)
3. **Congestion Level** (Estimated using lane occupancy metrics)
4. **Accident Alerts** (Detected using anomaly detection in vehicle movement)
5. **Helmet Violation Detection** (YOLO-based detection of riders without helmets)

### Compression Techniques
- **H.265** for efficient video compression
- **Blob/Base64** for metadata storage and transmission

### Connectivity
- **PoE+ (Power over Ethernet) for camera connectivity**
- **PoE Switch (₹8,000 per unit)**
- **PoE Splitter (₹3,000 per unit)**
- **Ethernet Cable (₹10 per meter, assumed 50m per junction)**

### Storage
#### On-Site Storage
- **NVR HDD:** ₹1,000 - ₹3,000 per TB (7-day retention)
- **NAS HDD:** ₹3,000 per TB (30-day retention)

#### Cloud Storage
- **AWS S3 Glacier:** ₹450 per TB (up to 1-year retention, for specific events)

### Smart Traffic Light Control
- Junctions adjust signal timing based on real-time traffic data.
- **Total cycle time: 12 minutes (720 seconds)**
- Each lane receives a **minimum 90 seconds**, and the remaining time is **dynamically allocated** based on congestion levels using **ST-GCN** (Spatio-Temporal Graph Convolutional Network) for short-term predictions.

### Forecasting & Optimization
- **Short-Term Prediction (ST-GCN, Next 10–30 min):** Helps dynamically adjust future signal timings.
- **Long-Term Prediction (DCRNN, Next few hours/days):** Predicts traffic trends to optimize city planning and road infrastructure.

### Machine Learning Models Used
#### **ST-GCN (Spatio-Temporal Graph Convolutional Network)**
- **Why ST-GCN?** It is effective in handling spatio-temporal relationships in road networks.
- **Working Principle:**
  - Represents road junctions and vehicle flow as a **graph**.
  - Uses **graph convolutional layers** to model spatial dependencies.
  - Employs **temporal convolutions** to capture time-based traffic variations.
  - Enables **real-time forecasting** (next **10-30 minutes**) for dynamic signal adjustments.

#### **DCRNN (Diffusion Convolutional Recurrent Neural Network)**
- **Why DCRNN?** It excels in long-term traffic forecasting (hours to days ahead).
- **Working Principle:**
  - Uses a **diffusion process** over a road network graph to capture traffic patterns.
  - **Recurrent layers** (LSTMs/GRUs) model temporal dependencies.
  - Helps in optimizing long-term traffic policies, roadwork planning, and congestion mitigation.

## Budget Estimation

### Cost Per Junction (Assuming 4 Cameras per Junction)
| Component           | High-Traffic (₹) | Mid-Traffic (₹) |
|--------------------|----------------|----------------|
| 4 Cameras         | 1,80,000        | 48,000         |
| Jetson Xavier NX  | 45,000          | 45,000         |
| PoE Switch       | 8,000           | 8,000          |
| PoE Splitter     | 3,000           | 3,000          |
| Ethernet (50m)   | 500             | 500            |
| NVR HDD (7 Days) | 3,000           | 3,000          |
| NAS HDD (30 Days)| 3,000           | 3,000          |
| Cloud (Per TB)   | 450             | 450            |
| **Total**        | **2,42,950**      | **1,10,950**     |

### Centralized System Cost
| Component          | Estimated Cost (₹) |
|-------------------|------------------|
| Centralized Control Server | 10,00,000 |
| Data Processing Server     | 8,00,000  |
| Cloud Storage Setup       | 5,00,000  |
| Network Infrastructure     | 4,00,000  |
| **Total**                  | **27,00,000** |

### Overall Cost
#### Total Cost for All Junctions
| Category         | Count | Cost Per Junction (₹) | Total Cost (₹) |
|----------------|------|---------------------|--------------|
| High-Traffic   | 50   | 2,42,950             | 1,21,47,500   |
| Mid-Traffic    | 345  | 1,10,950             | 3,82,77,750   |
| **Total**      | 395  | -                   | **5,04,25,250** |

#### Grand Total (Including Centralized System)
**₹5,04,25,250 + ₹27,00,000 = ₹5,31,25,250**

## Conclusion
This **AI-powered traffic management system** will significantly enhance urban mobility by:
- **Real-time monitoring** with **YOLOv8-based analytics**.
- **Dynamic traffic signal control** using **ST-GCN & DCRNN**.
- **Efficient congestion forecasting** to prevent future jams.
- **Optimized storage solutions** with hybrid on-site and cloud-based retention.

This scalable approach ensures a **smarter, safer, and more efficient** urban traffic system for Lucknow.



![image](https://github.com/user-attachments/assets/d1390ac9-50f0-47ae-8ae7-55ce3d20d619)
![image](https://github.com/user-attachments/assets/32a7bf9e-7605-43df-94a3-9d0c3c90c2fe)
![image](https://github.com/user-attachments/assets/56eae6a2-e589-47cc-8ca9-0bf7f8e17859)
![image](https://github.com/user-attachments/assets/0b194a8a-f7cd-4344-8caa-a251665d9ac1)
![image](https://github.com/user-attachments/assets/ccd8bbbd-43ab-4e0e-bdc8-e0d94ede6c90)
![image](https://github.com/user-attachments/assets/be72a16e-37e4-43f4-a87d-a51b1871b6db)
![image](https://github.com/user-attachments/assets/e1690228-d412-4759-8f6a-7e3ce314119c)
![image](https://github.com/user-attachments/assets/1f3417ac-d2e1-4e40-8d85-7511887af2b4)

