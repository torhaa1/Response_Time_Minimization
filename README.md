# Response Time Minimization for Police Units in Oslo

## Overview
This repository explores optimization problems in the context of spatial network analysis.
It specifically focus on the Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) approaches to minimize response times for police cars to events across the road network in Oslo, Norway.
Results can be used to strategically position police units better to ensure rapid response, thereby enhancing public safety and operational efficiency.


## Project Structure

### Key Components
1. **Data Loading**: Mainly OSMnx for importing/manipulating road networks from OpenStreetMap.
2. **Event Point Simulation**: Based on available data on population, crime rates, etc, to identify potential event nodes.
3. **Candidate Point Selection**: Sampling + filtering process to narrow down viable police car locations.
4. **Effective Computation**: Utilize fast libraries and methods for heavy tasks, like computing Cost Matrix.
5. **Problem Formulation and Solution**: Modeled as an LP/MILP problem using PuLP.
6. **Visualization**: Static and interactive maps to visualize optiaml police unit locations and resulting response times.

### Iterative scaling of problem size
- [x] Initial exploration and pipeline building: Grünerløkka district in Oslo.
- [x] Testing on medium sized networks: Whole of Oslo municipality.
- [ ] Max size test: National scale (Norway) and potentially other countries.

## High-level roadmap
- [x] **Solving the static case**: The initial phase will focus on solving the static case, where the events are fixed and the police units are strategically positioned to minimize response times. This can be used to explore parameters like the number of police units, their locations, and the impact of different response time thresholds.
- [x] **Visualizing the results**: The results will be visualized using static and interactive maps to provide intuitive insights into the effectiveness of proposed solutions. Isochrones will be used to visualize the areas that can be reached within a certain time frames (could represent legal response time thresholds).

### Further exploration can include:
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

### Optimize locations for Police Units in Oslo

#### Based on a sample of possible police car locations and a set of simulated events,  find the optimal locations for 4 police units to minimize the total response time to all events.

Figure 1 shows the solution with optimal placement of 4 units and their assigned events.   
<img src="https://github.com/torhaa1/Response_Time_Minimization/blob/main/figures/oslo_4locations_optimal.png" width="700">

Figure 2 plot the isochrones, showing how far one can reach from the optimal locations in 2, 4 and 8 minutes.
<img src="https://github.com/torhaa1/Response_Time_Minimization/blob/main/figures/oslo_4locations_isochrones.png" width="700">
