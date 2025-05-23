# AI for Unmanned Systems

This repository contains my COMPE 696 *AI for Unmanned Systems* programming assignments, covering neural-network modeling, PID control for quadrotors, and path-planning algorithms.

## Assignments

- **Assignment 1** – **Neural Networks on MNIST**  
  Python code implementing:  
  - Two simple neural nets with layer sizes (512, 10) and (1024, 10)  
  - Two deep nets with layer sizes [1000, 100, 10] and [1024, 512, 256, 128, 10]  
  Trains each on the MNIST dataset, logs training vs. testing accuracy, and plots cost vs. epoch curves for four cases :contentReference[oaicite:0]{index=0}.

- **Assignment 2** – **PID Controller Tuning for UAV**  
  Python script (`drone_pid.py` and helpers) that:  
  - Implements a PID controller for a quadrotor in X, Y, and Z axes  
  - Tunes proportional, integral, and derivative gains to minimize overshoot and settling time  
  - Generates convergence plots and discusses simulator setup/debugging challenges :contentReference[oaicite:1]{index=1}.

- **Assignment 3** – **Grid-Based Path Planning**  
  Python implementations of:  
  - **Dijkstra’s algorithm** for uniform-cost search  
  - **A\*** with an admissible heuristic for faster goal-directed search  
  Visualizes obstacle maps, overlays computed shortest paths, and compares node expansions and runtime trade-offs :contentReference[oaicite:2]{index=2}.
