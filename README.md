# Aurora

A modular solar array simulation software for Longhorn Racing Solar.  
It models PV cells, substrings, strings, and arrays, simulates MPPT algorithms, and provides desktop UIs for interactive analysis.  

---

## Features
- **Physics-based modeling**
  - Single-diode PV cell model with temperature and irradiance dependencies  
  - Hierarchical design: Cell → Substring → String → Array  

- **MPPT Algorithms**
  - Perturb & Observe (P&O)  
  - Incremental Conductance (IncCond)   

- **Simulation Engine**
  - Fixed timestep loop
  - Real-time or fast-forward modes  
  - Thread-safe messaging between simulation and UI  
  - Data is stored in data/runs

- **Interactive UI**
  - Desktop (PyQt6 + Plotly) for live IV/PV curves, efficiency metrics, and controls  

---

## Installation

Clone the repository:
```bash
git clone https://github.com/<your-username>/Power-Generation-Eclipse-SW.git
cd Power-Generation-Eclipse-SW
```

---

## Documentation

The code architecture and design document is on the [Confluence](https://cloud.wikis.utexas.edu/wiki/spaces/LHRSOLAR/pages/486541418/2024+-+2026+Simulation+Software)