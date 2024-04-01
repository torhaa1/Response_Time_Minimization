# Response Time Minimization for Police Units in Norway

## Overview

This repository explores the location optimization of police units on the road network to minimize response time to events. The analysis is performed separately for each police district.
The problem is a Mixed-Integer Linear Programming (MILP) problem defined and solved using PuLP. 

*A MILP problem involves finding the best solution from a finite set of possible solutions to a problem that includes both linear relationships and integer constraints.  It's ideal for situations like this, where the goal is to determine the optimal placement of police units (discrete choices) while minimizing response times (a continuous outcome). PuLP is a powerful Python library, offering an intuitive and efficient way to define and solve such problems.*

Given the start parameters (number of cars, number of events, car capacity, travel time cost matrix), the PuLP solution is returned, and the optimal car locations with the assigned events to each police car are plotted. The optimal car locations with assigned events can be used to model the expected response time for each police district. 

Statistics are computed for the individual police cars, as well as the performance of the police districts as a whole. Individual car statistics provide insights into strategically locating units within a district with respect to response time, shared workload, etc. District statistics include mean and median response time, which can be compared between districts, highlighting which districts are in most need of more resources to meet an acceptable (modeled) response time.

## Core computational challenge:  
MILP problems are inherently NP-hard because they require a search through a finite, but often vast, set of potential solutions where the decision variables can be any integer. 
NP-hardness indicates that while verifying the feasibility of a given solution is straightforward, discovering the best possible solution is computationally intensive and becomes exponentially harder as the problem size increases.   
This is becomes particularly problematic for large road networks with numerous potential car locations and events. 

### Example:  
For a reasonably sized MILP problem, let's consider 10 potential car locations and 1000 simulated events resulting in a cost matrix of `10,000 rows` (every car-to-event combination). Imagine this takes just 1 minute to compute on a standard laptop. However, if we increase the precision by using 50 potential car locations and 5,000 simulated events, the cost matrix explodes to a whopping `250,000 rows`. 

This represents a 25-fold increase in problem size, and the computation time would likely increase by thousands of times compared to the smaller first example due to the exponential complexity of MILP problems.
This highlights the importance of problem reduction and effective modeling techniques for road network problems before solving the MILP problem. Otherwise, using traditional MILP approaches might become entirely impractical for large road network optimization.

## Problem reduction and efficient network processing
Several strategies were implemented to speed up processing and reduce the problem size before solving the MILP problem:

- Use fast python libraries written in C (pandana, igraph) for intensive tasks like computing cost matrix and network centrality measures.
- Constrain simulated event points and potential car locations to populated areas (leveraging correlation between crime rates and population density)
- Filter potential car points based on network centrality and proximity to other car locations.
- Using dictionary data structures for max iteration speed and PuLP processing.
- Cost matrix reduction by filtering out the highest travel times, assuming there likely is another closer car to the given event.

### Example of problem reduction:  
Original problem:  
We distribute 50 potential car locations and 1000 simulated events resulting in a cost matrix of `50,000 rows`.

With our strategy:  
- Cut the car locations in half (=25 locations), by filtering on centrality and proximity.
- Cut the cost matrix (25,000) in half, by filtering out top travel times.
- Resulting in a cost matrix of `12,500 rows` (25% the size of the original)

**Conclusion**: Faster libraries, smarter problem setup, and clever problem reductions makes using MILP for optimal solutions on large road networks more attractive.

---

## Project Structure

### Iterative upscaling of network size
- [x] Initial exploration and pipeline building: Grünerløkka district in Oslo.
- [x] Medium sized network: Oslo municipality/Oslo police district.
- [ ] Large sized networks: (11 police districts) Eastern, SouthEastern, SouthWestern, etc
- [ ] Max size network (if ignoring police districts): National scale (Norway).

## High-level roadmap
- [x] **Solving the static case**: The initial phase will focus on solving the static case, where the events are fixed and the police units are strategically positioned to minimize response times. This can be used to explore parameters like the number of police units, their locations, and the impact of different response time thresholds.
- [x] **Visualizing the results**: The results will be visualized using static and interactive maps to provide intuitive insights into the effectiveness of proposed solutions. Isochrones will be used to visualize the areas that can be reached within a certain time frames (could represent legal response time thresholds).

### Future directions that could be explored in different repos:
- [ ] **Dynamic and time-dependent variables**: Incorporating dynamic and time-dependent variables to adapt to changing urban dynamics. E.g. traffic congestion, time of day, population migration, etc.
- [ ] **Heuristic methods**: Exploring heuristic methods to solve the problem faster. Could be used for real-time decision making.
- [ ] **Simulations and scenario analysis**: Integrating more comprehensive simulations to explore different scenarios and their impact on response times.


## Project Development Log

Below is a detailed development log highlighting the progress and methodologies employed at various stages of the project:

| Notebook                         | Description |
| -------------------------------- | ----------- |
| `08_Grunerlokka_MCLP_v1.0.ipynb` | **Area:** Grünerløkka district (small network)<br>**Obj. func:** Maximize number of nodes covered within service distance<br>**Event points:** Generated based on population density + randomness<br>**Event weight:** Inverse distance^2 to 1 handpicked crime hotspot<br>**Possible car locations:** Handpicked 10 points<br>**Car service distance:** 800m<br>**Implementation:** OSMnx, NetworkX, spopt.MCLP function |
| `11_Grunerlokka_PuLP_v1.1.ipynb` | **Area:** Grünerløkka district (small network)<br>**Obj. func:** Minimize total response time<br>**Event points:** Generated based on population density + randomness (same)<br>**Event weight:** Defined distance intervals from 1 handpicked crime hotspot (not used in LP problem)<br>**Possible car locations:** Handpicked 10 points -> 2 in solution<br>**Car service distance:** 800m<br>**Car max capacity:** 200 events<br>**Implementation:** OSMnx, Pandana, PuLP |
| `12_Grunerlokka_PuLP_v1.2.ipynb` | **Area:** Grünerløkka district (small network)<br>**Obj. func:** Minimize total response time<br>**Event points:** Sample from original network nodes (fast for testing purposes)<br>**Event weight:** None<br>**Possible car locations:** Sampled 60 nodes -> Filtered to 34 -> 4 in solution<br>**Car service distance:** 800m<br>**Car max capacity:** 300 events<br>**Implementation:** OSMnx, NetworkX, Pandana, PuLP, Descartes<br>**Computation time:** Node centrality (4s), PuLP add constraints (1min 16sec) + solve (1min 14sec) |
| `13_Oslo_PuLP_v2.0.ipynb`   | **Area:** Oslo municipality, minus Nordmarka (medium network)<br>**Obj. func:** Minimize total response time<br>**Event points:** Sample from original network nodes (610 nodes)<br>**Event weight:** None<br>**Possible car locations:** Sampled 400 nodes -> Filtered to 104 -> 4 in solution<br>**Car service distance:** 800m<br>**Car max capacity:** 300 events<br>**Implementation:** OSMnx, NetworkX, Pandana, PuLP, Descartes<br>**Computation time:** Node centrality (4min 6sec), PuLP add constraints (6min 18sec) + solve (4min 38sec) |


## Example Use Case

### Optimize locations for 4 Police Units in Oslo City

#### Based on a sample of possible police car locations and a set of simulated events,  find the optimal locations for 4 police units to minimize the total response time to all events.

<img src="https://github.com/torhaa1/ResponseTimeMinimization/blob/main/figures/oslo_4locations_optimal.png" width="700">
Figure 1 shows the solution with optimal placement of 4 units and their assigned events.   

<img src="https://github.com/torhaa1/ResponseTimeMinimization/blob/main/figures/oslo_4locations_isochrones.png" width="700">
Figure 2 plot the isochrones, showing how far one can reach from the optimal locations in 2, 4 and 8 minutes.
