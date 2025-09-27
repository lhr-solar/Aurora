# Aurora

A modular solar array simulation software for Longhorn Racing Solar (2024–2026).  
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
  - Fixed timestep loop (default 100 Hz)  
  - Real-time or fast-forward modes  
  - Thread-safe messaging between simulation and UI  

- **Interactive UI**
  - Desktop (PyQt6 + Plotly) for live IV/PV curves, efficiency metrics, and controls  
  - Web (Dash/Streamlit) for lightweight visualization  

- **Reproducibility**
  - Config-driven (`configs/default.yaml`)  
  - Per-run logs, metrics, and plots auto-saved in `data/runs/`  

---

## Installation

Clone the repository:
```bash
git clone https://github.com/<your-username>/Power-Generation-Eclipse-SW.git
cd Power-Generation-Eclipse-SW
```

---

## Usage

After cloning and installing dependencies (see `requirements.txt`), you can run the simulator from the `simulators` package or open the desktop UI.

Example (run sim):
```bash
python -m simulators.mppt_sim
```

You can add additional sections after the closed code block. For example, add `## Configuration` or `## Development` as needed.
