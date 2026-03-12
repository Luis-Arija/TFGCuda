Overview

This project implements different approaches to simulate the N-Body gravitational problem using both CPU and GPU (CUDA). It was developed as part of my Bachelor's Thesis on parallelizing irregular algorithms on GPU architectures.
The goal of the project is to study how parallel computing on GPUs can improve performance and how irregular algorithms, such as spatial tree structures, behave compared to regular algorithms.

Several versions of the simulation were implemented:

1. A CPU version using the direct N² algorithm
2. A GPU CUDA implementation of the same algorithm
3. A QuadTree spatial partition algorithm
4. A GPU CUDA QuadTree spatial partition algorithm
5. Hybrid CPU + GPU versions of the QuadTree simulation
6. A Python visualization tool to animate the simulation results


The N-Body Problem:
The N-Body problem simulates the movement of particles that interact through gravitational forces.
Each particle is influenced by every other particle using Newton’s law of gravitation:

F = G * (m1 * m2) / r²

For every timestep the simulation calculates:
-Gravitational forces
-Acceleration
-Velocity
-New position

The straightforward approach requires calculating all pairwise interactions, giving a complexity of O(N²)
This becomes very expensive when the number of bodies grows, which makes the problem suitable for parallel computing with GPUs.

Implementations
011 - CPU Direct Algorithm
This version runs entirely on the CPU and calculates the force between every pair of bodies.
It serves as a baseline implementation to compare performance with the GPU versions.

Characteristics:
-Exact calculation
-Simple implementation
-Poor scalability for large numbers of bodies

012 - CUDA GPU Parallel Algorithm

This version parallelizes the N² algorithm using CUDA.
Each GPU thread computes interactions between bodies, allowing the simulation to run much faster than the CPU version when many bodies are present.

The simulation is split into several CUDA kernels:

-Force calculation
-Force aggregation
-Acceleration computation
-Velocity update
-Position update

021 - QuadTree Spatial Algorithm


To reduce the computational cost, a QuadTree spatial partitioning algorithm was implemented.
The simulation space is recursively divided into four regions, creating a hierarchical tree structure.
Bodies that are far away can be approximated using the center of mass of a region, reducing the number of calculations required.

Advantages:

Lower computational cost
Better scalability for large systems
Demonstrates an irregular algorithm structure

023 - Hybrid QuadTree CPU + GPU

Files:

quaTreeGPU.cu
quaTreeCPUGPU.cu

These versions combine CPU and GPU computation.
The CPU manages the tree structure and spatial partitioning, while the GPU performs the parallel force calculations.
This approach explores how irregular algorithms can be partially parallelized on GPUs.

Animates particle movement using FuncAnimation

Libraries used:

matplotlib

numpy
