# 3DP-CPTP-
Code for the project 3DP-CPTP 

DFJ subtour elimination constraints are much more efficient than MTZ subtour elimination constraints.

3D Loading has additional physical and operational constraints e.g. fragility, supporting area, LIFO policy 
=> our model is Packing not Loading

Based on Gendreau et al. 2006: A Tabu Search Algorithm for a Routing and Container Loading Problem

Locations, Demand by the customers, Vehicle weight capacities were taken from 27 Euclidean CVRP instances (see Toth and Vigo 2002).
Loading space was defined as W = 25, H = 30, and L = 60.
Number of requested items was uniformly and randomly generated between 1 and 3.
Each item, weight height and length uniformly randomly generated in the interval 0.2W-0.6W (resp. 0.2H-0.6H, resp. 0.2L-0.6L).

Analog to Caspar et al. 2026: Solution Methods for the Dynamic and Probabilistic Profitable Tour Problem

P1: All prizes (revenues) are equal to 100.
P2: Pseudo-random (r_i = 1 + ((7141·i+ 73) mod 100) for all i)
P3: Distance-based
P4: Proportional to the depot distance (r_i = c0,i for all i) 

-We use the HPC cluster Elwetritsch of the RPTU Kaiserslautern-Landau which consists of several AMD EPYC 9135 processors (Note that a single thread of the cluster is comparable with that of a standard desktop computer processor) - 1 cpu core and 1800 CPU seconds - in the paper we tested different parameters

TO DO:

-Improving COSA

