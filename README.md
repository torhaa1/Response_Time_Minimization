# Response Time Minimization for Police Units in Norway

## Introduction

This repository explores location optimization of police units within their police district to minimize response time to events. 
The analysis is performed separately for each police district. The problem is a Mixed-Integer Linear Programming (MILP) problem defined and solved using PuLP.

*A MILP problem finds the best solution from a finite set of possible solutions to a problem that includes both linear relationships and integer constraints.  Ideal for situations like this, where the goal is to determine the optimal placement of police units (discrete choices) while minimizing response times (a continuous outcome). PuLP is a powerful Python library, offering an intuitive and efficient way to define and solve such problems.*

## Repository content

### [**Preprocessing notebooks:**](https://github.com/torhaa1/ResponseTimeMinimization/tree/main/preprocessing)
The preprocessing scripts prepare and save data locally for fast loading in later analysis. It fetches the boundaries, road network (openstreetmap) and population data for the district. Edge speeds (speed limit) is added and network centrality measures are computed. An exploratory data analysis show the correlation between crime and population at the municipality level.

### [**Analysis notebooks:**](https://github.com/torhaa1/ResponseTimeMinimization/tree/main/main_analysis)
Main analysis is performed for each district in separate notebooks. Also contains the [statistics notebook](https://github.com/torhaa1/ResponseTimeMinimization/blob/main/main_analysis/43_district_statistics.ipynb) and [python module](https://github.com/torhaa1/ResponseTimeMinimization/blob/main/main_analysis/utilityModule.py) that holds the larger custom functions.
The number and location of simulated events placed in a police district follows the population data (250x250m gridcells).
Next, a polygon is drawn around the areas with high population density to constrain the area from where to sample possible car locations.
A set of car points is randomly sampled from the high density area and furthered filtered based on centrality measures (closeness and betweenness).
Next, the cost matrix with every origin-destination (car-to-event) pair is computed and the problem is defined and solved as a MILP problem using the PuLP library. Here the goal is to minimize the total response time while respecting the max workload/capacity for each car.

**Spatial Insight:**  
From the analysis results, we create the following plots:
- Optimal car locations along with their assigned events on the map.
- Isochrones showing the cars collective reach from the optimal car locations within 3 time intervals.
- Isochrone map as interactive leaflet map, but this only seem to work in some web browsers.

**Statistical Insight:**  
- The single district statistics can be used to give a high-level understanding of what is the best possible response time in the given district. It provides an objective baseline to measure historical data against. It is also possible to quantify the improvement/degradation in response time by adding/removing one or more cars.
- The between-district statistics provide a high-level understanding and an objective overview of the response time for the different districts nationally. Highlighting which districts are in most need of more resources to meet an acceptable (modeled) response time.
- The within-district statistics show each car's response time and workload, providing insights into strategically locating units better.


## Core computational challenge:  
MILP problems are inherently NP-hard because they require a search through a finite, but often vast, set of potential solutions where the decision variables can be any integer. 
NP-hardness indicates that while verifying the feasibility of a given solution is straightforward, discovering the best possible solution is computationally intensive and becomes exponentially harder as the problem size increases.   
This is becomes particularly problematic for large road networks with numerous potential car locations and events. 

### Example:  
For a reasonably sized MILP problem, let's consider 10 potential car locations and 1000 simulated events resulting in a cost matrix of `10,000 rows` (every car-to-event combination). Imagine this takes just 1 minute to compute on a standard laptop. However, if we increase the precision by using 50 potential car locations and 5,000 simulated events, the cost matrix explodes to `250,000 rows`. 

This represents a 25-fold increase in problem size, and the computational time would likely increase much more than that compared to the smaller first example due to the exponential complexity of MILP problems.
This highlights the importance of problem reduction and effective modeling techniques for road network problems before solving the MILP problem. Otherwise, using traditional MILP approaches might become entirely impractical for large road network optimization.

### Problem reduction and efficient network processing
Several strategies were implemented to speed up processing and reduce the problem size before solving the MILP problem:

- Use fast python libraries written in C (pandana, igraph) for intensive tasks like computing cost matrix and network centrality measures.
- Constrain simulated event points and potential car locations to populated areas (leveraging correlation between crime rates and population density)
- Filter potential car points based on network centrality measures.
- Using dictionary data structures for max iteration speed and PuLP processing.
- Cost matrix reduction by filtering out the highest travel times, assuming there likely is another closer car to the given event. Works well when population is spread around the network. Less applicable if population is clustered in one side of the network.

### Example of ideal problem reduction:  
Original problem:  50 potential car locations and 1000 simulated events results in a cost matrix of `50,000 rows`.   
The size of the cost matrix (problem size) is reduced by:
- Cut the car locations in half (=25 locations), by filtering on centrality measures.
- Cut the cost matrix (25,000) in half, by filtering out top travel times.
- Resulting in a cost matrix of `12,500 rows` (25% the size of the original)

---

## Example 1: Optimize locations for 4 Police Units in Oslo City
#### Based on a sample of possible police car locations and a set of simulated events,  find the optimal locations for 4 police units to minimize the total response time to all events, while respecting each car's maximum workload.

<img src="https://github.com/torhaa1/ResponseTimeMinimization/blob/main/figures/oslo_4locations_optimal.png" width="700">
Figure 1 shows the solution with optimal placement of 4 units and their assigned events.   


<img src="https://github.com/torhaa1/ResponseTimeMinimization/blob/main/figures/oslo_4locations_isochrones.png" width="700">
Figure 2 plot the isochrones, showing the cars collective reach from the optimal car locations within 2, 4 and 8 minutes.


## Example 2: Compare response time across all 12 districts

#### Using 4 cars per district, we can compare the modeled ideal response time (arrival of first car) between the police districts.   
Note that there is a large variation in population and area coverage between the districts.

<img src="https://github.com/torhaa1/ResponseTimeMinimization/blob/main/figures/4cars_between-district_stats.png" width="700">
Figure 3 show the main response time statistics between the districts, when using 4 cars per district.
