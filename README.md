# 3DP-CPTP

-Based on the data set of Gendreau et al. 2006: A Tabu Search Algorithm for a Routing and Container Loading Problem.
-DFJ subtour elimination constraints are much more efficient than MTZ subtour elimination constraints.
-Packing, since Loading has additional physical and operational constraints e.g. fragility, supporting area, LIFO policy. 
-Prizes are generated analogously to Caspar et al. 2026: Solution Methods for the Dynamic and Probabilistic Profitable Tour Problem.
  P1: All prizes (revenues) are equal to 100.
  P2: Pseudo-random (r_i = 1 + ((7141·i+ 73) mod 100) for all i)
  P3: Distance-based
  P4: Proportional to the depot distance (r_i = c0,i for all i) 
-Experiments are made on the HPC cluster Elwetritsch of the RPTU Kaiserslautern-Landau, which consists of several AMD EPYC 9135 processors (Note that a single thread of the cluster is comparable with that of a standard desktop computer processor) 
