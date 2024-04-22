# Response Time Minimization for Police Units in Norway

## Introduction
This repository explores location optimization of police units within their respective police district, with the aim to minimize response time to incidents. The analysis is performed separately for each police district. The challenge is framed as a MILP problem and is defined and solved using the PuLP library.

*A Mixed-Integer Linear Programming (MILP) problem identifies the optimal solution from a finite set of possibilities, with both linear relationships and integer constraints. This is suitable for our scenario, where the objective is optimal placement of police units (discrete choices) while minimizing response times (a continuous outcome). The PuLP Python library, offers an intuitive and efficient way to define and solve such problems.*

## Repository content

### [**Preprocessing notebooks**](https://github.com/torhaa1/ResponseTimeMinimization/tree/main/preprocessing)
The preprocessing notebooks prepare and save data locally for fast loading in later analysis.  Fetches boundaries, road network (OpenStreetMap) and gridded population data (250x250m; from ssb.no) for each district. Edge speeds (speed limits) is added and network centrality measures are computed. An exploratory data analysis highlights the strong correlation between crime and population at the municipality level, justifying the exclusive use of population data for generating crime events.

### [**Analysis notebooks**](https://github.com/torhaa1/ResponseTimeMinimization/tree/main/main_analysis)
The main analysis is performed for each district in separate notebooks. This folder also include the [statistics notebook](https://github.com/torhaa1/ResponseTimeMinimization/blob/main/main_analysis/43_district_statistics.ipynb) and [python module](https://github.com/torhaa1/ResponseTimeMinimization/blob/main/main_analysis/utilityModule.py) that contains the larger custom functions. The number and locations of simulated events within a police district are based on the gridded population data. A polygon is then drawn around areas with high population density to define the sampling area for potential car locations. From this area, car points are randomly sampled and further filtered using centrality measures (closeness and betweenness). Then, a cost matrix for every origin-destination (car-to-event) pair is created. The problem is defined and solved as a MILP problem, aiming to minimize the total response time while adhering to the maximum workload for each car.

**Spatial Insight**  
Based on the analysis results, we generate the following visualizations:
- A map displaying the optimal car locations along with their assigned events.
- Isochrones illustrating the collective reach of the cars from these optimal locations within 3 time intervals.
- An interactive isochrone map using Leaflet, though Github has problems visualizing it in some web browsers.

**Statistical Insight**  
- **Single District Statistics:** These statistics illustrate the best possible response times for a district, with a set number of cars allocated. They serve as an objective baseline for comparing against historical data. Additionally, changes in modeled response time can be quantified by adding or removing cars from the district.
- **Between-District Statistics:** These provide a high-level overview and an objective comparison of modeled response times across different districts nationally. It highlights which districts that are in most need of more resources to meet an acceptable modeled response time.
- **Within-District Statistics:** These statistics shows the response time and workload for each car, providing valuable insights for strategically positioning units more effectively.

#### Model Assumptions on Response Time
This model calculates the best possible scenario for response times by using the speed limit for each road segment as the base speed. Police vehicles are assumed to drive 20-36% faster than this speed limit. E.g., speeds are increased by 22% for roads with a 20 kph limit and by 36% for roads with a 110 kph limit.
It is important to note that the model does not account for potential decreases in travel speed due to external factors such as poor weather conditions, seasonal variations, or heavy traffic. Excluding these elements means that the model presents an idealized scenario, focusing on the theoretical minimum response time under optimal conditions. Additionally, the model does not allow the use of ferries to reduce drive time.



## Core computational challenge 
MILP problems are classified as NP-hard because they require a search through a finite, but often vast, set of potential solutions where the decision variables are integers. NP-hardness indicates that while verifying the feasibility of a given solution is straightforward, discovering the best possible solution is computationally intensive and becomes exponentially harder as the problem size increases. This complexity can become particularly problematic for large road networks, where there are numerous potential locations for police cars and events.

### Example
For a reasonably sized MILP problem, let's consider 10 potential car locations and 1000 simulated events resulting in a cost matrix of `10,000 rows` (every car-to-event combination). Imagine this takes just 1 minute to compute on a standard laptop. However, if we increase the precision by using 50 potential car locations and 5,000 simulated events, the cost matrix explodes to `250,000 rows`. 

This represents a 25-fold increase in problem size, and the computational time would likely increase much more than that due to the exponential complexity of MILP problems.
This highlights the importance of problem reduction and effective modeling techniques for road network problems before solving the MILP problem. Otherwise, using traditional MILP approaches might become entirely impractical for large road network optimization.

### Problem reduction and efficient network processing
Several strategies were implemented to speed up processing and reduce the problem size before solving the MILP problem:

- Use fast python libraries written in C (pandana, igraph) for intensive tasks like computing cost matrix and network centrality measures.
- Constrain sampling of car locations to most populated areas.
- Filter potential car locations based on network centrality measures.
- Using dictionary data structures for max iteration speed and PuLP processing.
- Cost matrix reduction by filtering out the highest travel times, assuming there likely is another closer car to the given event. Works well when population is spread around the network. Less applicable if population is clustered in one side of the network.

### Example of ideal problem reduction
Original problem:  50 potential car locations and 1000 simulated events results in a cost matrix of `50,000 rows`.   
The size of the cost matrix (problem size) is reduced by:
- Cut the car locations in half (=25 locations), by filtering on centrality measures.
- Cut the cost matrix (25,000) in half, by filtering out top travel times.
- Resulting in a cost matrix of `12,500 rows` (25% the size of the original)

---

## Example 1: Optimize locations for 4 Police Units in Oslo City
This example demonstrates how to find the optimal locations for 4 police units in Oslo City. The goal is to minimize the total response time to all events while adhering to each car's maximum workload.

<img src="https://github.com/torhaa1/ResponseTimeMinimization/blob/main/figures/oslo_city_4cars_plot_locations.png" width="700">
Figure 1 shows the solution with optimal placement of 4 units and their assigned events.   

<img src="https://github.com/torhaa1/ResponseTimeMinimization/blob/main/figures/oslo_city_4cars_plot_isochrones.png" width="700">
Figure 2 plot the isochrones, illustrating the cars collective reach from the optimal car locations within the time intervals of 2, 4 and 8 minutes.


## Example 2: Compare response time across all 12 districts
This example use 4 cars per district to compare the modeled ideal response times (arrival of first car) across all police districts. It’s important to note that there is a significant variation in population density and area coverage among the districts.

<img src="https://github.com/torhaa1/ResponseTimeMinimization/blob/main/figures/4cars_between-district_stats.png" width="700">
Figure 3 show the main response time statistics between the districts, when using 4 cars per district.
