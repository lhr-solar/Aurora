###Things that need to be made better###
- Figure out why numbers we’re getting isn’t similar to real world

###True GMPP Oracle Mode###
- Currently benchmark against gmpp per tilmestep
- Instead compute global mpp surface across time
- Store full PV curves per tilmestep
- Calculate: tracking eff %, energy harvested vs theoretical max, settling time, oscillation amplitude, recovery time after shading events
- Add a tracking metrics panel that auto calculates η_tracking = ∑ P_algo(t) / ∑ P_GMPP(t)

###Shading Pattern Generator###
- Right now rely on sliders and profiles
- Add a parametric shading generator
    - Moving cloud model (sinusoidal attenuation)
    - Step shading (partial bypass activation)
    - Randomized string mismatch
    - Fault injection (dead substring)
- Makes Aurora be able to simulate: partial shading conditions, mismatch dynamics, bypass diode activation transitions

###Thermal Feedback Model###
- Right now temp is an input
- Add:
    - Temp as function of irradiance + current + ambient
    - Simple thermal RC model
- Introduces coupling: high current -> heat -> lower eff -> shifting MPP

###Headless Research Mode###
- Be able to run the sim with no UI
- Ex: python Aurora.py --benchmark all --profile iso.csv --runs 100 --export results.csv
- Capable of doing massive algo sweeps, Monte Carlo shading experiments, research reproducibility, faster batch evaluation

###Algo Plug-in System###
- Dynamically register via decorator: 
    - @register_mppt("pso") \ class PSOController(BaseMPPT):
- Then UI auto-populates from registry
- So that anyone can drop in a new algo file and it works

###Event Timeline Panel###
- Show bypass activation, profile change, irradiance spike, temp spike, algo mode switch, GMPP shift event

###Energy Accumulation Graph###
- E(t) = ∑ P(t) Δt
- Changes perspective from tracking to harvest outcome

###Hardware in the Loop Ready Interface###
- Add engine abstraction layer, real panel mode, simulated panel mode, to give it hardware input driver functionality

###ML-Based MPPT###
- Reinforcement learning MPPT, Model Predictive MPPT, LSTM-based predictor

###Stability Analysis Panel###
- For each algo, detect limit cycles, measure oscillation frequency, estimate small-signal stability, compute control gain sensitivity

###Publish Ready Stuff###
- MkDocs
- Example benchmark results
- Published “Aurora Benchmark Suite v1.0”
- Reproducibility protocol
