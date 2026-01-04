# Projected MPPI: Real-Time Safe Control via GPU-Accelerated Batched ADMM

## Overview

Projected MPPI is a high-performance control framework that bridges the gap between sampling-based control (MPPI) and optimization-based control (MPC). This repository implements a **GPU-accelerated Model Predictive Path Integral (MPPI)** controller that strictly enforces hard constraints by projecting sampled trajectories onto safe sets using a custom **Batched Alternating Direction Method of Multipliers (ADMM)** solver.

Built on **JAX**, the entire pipeline is JIT-compiled to a single GPU kernel, enabling the solution of thousands of Quadratic Programming (QP) and Conic Programming problems in parallel at rates exceeding 50Hz. This framework is model-agnostic and has been validated on both **Uni-cycle (2D)** and **Quadrotor (3D)** dynamics.

## Simulation Results

### 1. Drone Corridor Navigation (3D)
The drone navigates a constrained tunnel while strictly adhering to velocity and spatial boundary constraints. The solver corrects the stochastic MPPI samples in real-time.

| Drone Obstacle Avoidance (3D) | Drone Tunnel (3D) |
| :---: | :---: |
| <img src="drone/drone_obstacle.gif" width="100%"/> | <img src="drone/drone_corridor.gif" width="100%"/> |

### 2. Uni-cycle Narrow Tunnel (2D)
A non-holonomic vehicle navigates a Narrow Tunnel. Baseline method 

| Ours | Baseline |
| :---: | :---: |
| <img src="corridor_test/our_result.gif" width="100%"/> | <img src="corridor_test/baseline_result.gif" width="100%"/> |

## Theoretical Foundation & Extensibility

### Generalized Constraint Handling via Operator Splitting
A key contribution of this framework is the decoupling of dynamics satisfaction and constraint enforcement through ADMM's operator splitting mechanism. While the primary implementation focuses on Box constraints, the solver architecture is mathematically designed to handle a broad class of convex and non-convex constraints by simply substituting the projection operator ($z$-update step).

The Batched ADMM solver iterates between:
1.  **x-update (Dynamics):** Solving the linear system $Ax=b$ (Physics).
2.  **z-update (Constraints):** Projecting the solution onto constraint sets $\mathcal{C}$.

This modular structure allows for the seamless integration of:
* **Box Constraints (Implemented):** For state/input bounds (e.g., $u_{min} \le u \le u_{max}$) and linear corridor boundaries.
* **Second-Order Cone (SOC) Constraints (Supported Theory):** For constraints involving vector norms, such as maximum thrust limits ($\|u\|_2 \le T_{max}$) or friction cones in legged locomotion.
* **Non-Convex Geometric Constraints:** For obstacle avoidance (e.g., spherical or cylindrical exclusion zones) via geometric projection operators.

### B-Spline Parameterization
Trajectories are parameterized using B-Splines. By exploiting the **Convex Hull Property** of B-Splines, continuous-time safety constraints are enforced by strictly constraining only a finite set of control points, reducing the dimensionality of the optimization problem without sacrificing safety guarantees.

## Methodology

### Control Architecture
1.  **Sampling:** Thousands of control inputs are sampled based on the current nominal input distribution.
2.  **Linear Approximation:** The nonlinear dynamics are linearized around the current operating point to form the constraint matrices.
3.  **Residual Feedback Correction:** The mismatch between the nonlinear rollout and the linearized model is calculated as a residual term. This term is fed back into the ADMM solver to compensate for linearization errors, ensuring robust performance during aggressive maneuvers.
4.  **Batched Projection:** All samples are projected in parallel onto the feasible sets defined by the constraint operators.
5.  **Weighting & Update:** The projected (safe) trajectories are evaluated using the nonlinear model, and the optimal control input is updated via importance sampling.

### The Batched ADMM Solver
* **Parallelism:** Utilizes JAX's `vmap` to solve 4,000+ optimization instances concurrently on the GPU.
* **Caching:** Exploits the structure of the optimal control problem where the system matrix ($A$) is shared across samples. Cholesky factorization ($LL^T$) is computed once per timestep and reused, drastically reducing computational overhead.
