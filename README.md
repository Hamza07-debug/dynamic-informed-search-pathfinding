# Dynamic Informed Search Pathfinding Visualizer (Pygame)

A professional, interactive, and modular visualization system for **Informed Search Algorithms**, developed as part of the **Artificial Intelligence** university course.  
This project demonstrates **A\*** and **Greedy Best-First Search (GBFS)** in a **dynamic grid environment** with real-time obstacle handling and performance metrics.

---

## 📌 Project Overview

This application simulates a grid-based environment where an intelligent agent navigates from a **start node** to a **goal node** using informed search strategies.  
The system supports **dynamic re-planning**, allowing the agent to adapt to newly appearing obstacles during execution without restarting the entire search.

The project emphasizes:
- Correct algorithmic implementation
- Clean object-oriented design
- Real-time visualization
- Academic-level code quality

---

## ✨ Features

### Environment
- User-defined grid size (rows × columns)
- Fixed and clearly visible **Start** and **Goal** nodes
- Random maze generation with adjustable obstacle density
- Interactive map editor (add/remove walls using mouse)
- No static `.txt` map files

### Search Algorithms
- **A\*** Search: `f(n) = g(n) + h(n)`
- **Greedy Best-First Search (GBFS)**: `f(n) = h(n)`

### Heuristics
- Manhattan Distance
- Euclidean Distance  
(Selectable at runtime via GUI)

### Dynamic Mode
- Obstacles may appear randomly while the agent is moving
- If the current path becomes blocked:
  - The agent detects the obstruction
  - Re-plans efficiently from its current position
  - Avoids unnecessary full search restarts

### Visualization
- Frontier nodes → Yellow
- Visited nodes → Blue / Red
- Final path → Green
- Start node → Cyan
- Goal node → Purple
- Smooth animated movement of the agent

### Metrics Dashboard
- Total nodes expanded
- Final path cost
- Execution time (milliseconds)
- Obstacle density

---

## 🧠 System Architecture

The codebase follows a clean **object-oriented architecture**:

- `Grid` – Manages grid layout and obstacles  
- `Node` – Represents individual cells with cost values  
- `SearchAlgorithm` – Abstract logic for informed search  
- `AStar` / `GBFS` – Algorithm-specific implementations  
- `Agent` – Executes movement and dynamic re-planning  
- `Renderer` – Handles all GUI and visualization logic  

Search logic is fully separated from rendering to ensure clarity and maintainability.

---

## 🎮 Controls

| Action | Control |
|------|--------|
| Toggle wall | Left Mouse Button |
| Set Start node | Shift + Left Click |
| Set Goal node | Ctrl + Left Click |
| Start search | Space |
| Reset search | R |
| Clear obstacles | C |
| Increase / Decrease density | [ / ] |

---

## 🛠 Installation

### Requirements
- Python 3.13.5+
- Pygame

### Install Dependencies
```bash
pip install -r requirements.txt
