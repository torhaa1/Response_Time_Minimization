# Utility module to hold functions that are used in the main analysis scripts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping
from shapely.ops import nearest_points, unary_union
from descartes import PolygonPatch
import geopandas as gpd
import osmnx as ox
import networkx as nx
import pulp
from scipy.stats import gaussian_kde
import folium


#######################################################################
# PREPROCESSING
#######################################################################

# Function to increase edge speed and recompute travel time
def increase_edge_speeds(graph_gdf_edges):
    """
    Increase the speed of edges in the graph GeoDataFrame and recompute the travel time.
    Assume police cars can drive 20-40% faster than speed limits.
    Based on 3rd degree polynomial function described in issue #8.
    """
    # Use precomputed polynomial coefficients
    precomputed_coefficients = [-2.32614692e-05, 6.92679011e-03, 8.36197811e-01, 5.20808653e+00]
    polynomial = np.poly1d(precomputed_coefficients)
    
    # Apply the polynomial to adjust the speeds
    graph_gdf_edges['speed_kph'] = polynomial(graph_gdf_edges['speed_kph'])
    
    # Calculate new travel time in seconds: length (m) / (speed (km/h) / 3.6 (km/h to m/s))
    graph_gdf_edges['travel_time'] = (graph_gdf_edges['length'] / (graph_gdf_edges['speed_kph'] / 3.6))
    return graph_gdf_edges


#######################################################################
# EVENT POINTS
#######################################################################

# Function to simulate event points based on population count
def simulate_event_points(population_gdf, min_range=2000, max_range=2500):
    """
    Simulates the number of event points in each gridcell based on its population count.
    Use a binary search algorithm to find the optimal population multiplier to achieve a target number of points.
    Target range for the total number of points is defined by [min_target, max_target].    
    """
    # Define the target range for the total number of points
    min_target = min_range
    max_target = max_range
    
    # Initial lower and upper bounds for population_multiplier
    low = 0.001
    high = 0.1
    
    # Convergence threshold and maximum iterations to prevent infinite loops
    epsilon = 1e-6
    max_iterations = 100
    iteration = 0
    while iteration < max_iterations:
        mid = (low + high) / 2
        
        # Calculate the number of points using the current guess of population_multiplier
        population_gdf['num_points'] = np.round(mid * population_gdf['population']).astype(int)
        population_gdf['num_points'] = np.maximum(population_gdf['num_points'], 0) # Ensure non-negative
        total_points = population_gdf['num_points'].sum()
        
        # Check if the total number of points is within the target range
        if min_target <= total_points <= max_target:
            print(f"Total number of simulated event points: {total_points}. Target range [{min_target}, {max_target}], using population multiplier: {mid}")
            break
        elif total_points < min_target:
            low = mid
        else:
            high = mid
        
        # Check for convergence
        if abs(high - low) < epsilon:
            print(f"Convergence reached with multiplier: {mid} (total points: {total_points})")
            break
        iteration += 1
    
    if iteration == max_iterations:
        print(f"Max iterations reached with multiplier: {mid} (total points: {total_points})")
    return population_gdf


# Function to generate points within a grid cell
def generate_points_within_gridcell(num_points, bounds):
    """
    Generate points within given bounds.
    """
    min_x, min_y, max_x, max_y = bounds
    xs = np.random.uniform(min_x, max_x, num_points)
    ys = np.random.uniform(min_y, max_y, num_points)
    points = [Point(x, y) for x, y in zip(xs, ys)]
    return points


# Function to generate polygon around high-density areas of event points
def generate_high_density_polygon(event_points_gdf, grid_size=100, density_threshold="median", simplification_tolerance=10.0, plot_results=False):
    """
    Generates a polygon covering high-density areas based on event points.

    Parameters:
    - event_points_gdf: GeoDataFrame containing event points with 'geometry' column.
    - grid_size: Size of the grid for KDE calculation.
    - density_threshold: Density threshold to identify high-density areas. Can be 'median', 'mean', or a specific value.
    - simplification_tolerance: Tolerance for geometry simplification to ensure validity.
    - plot_results: If True, plots the KDE, high-density areas, and the final polygon.

    Returns:
    - A GeoDataFrame with a single polygon covering high-density event areas.
    """
    # Extract x and y coordinates of the points
    x = event_points_gdf.geometry.x
    y = event_points_gdf.geometry.y

    # Perform Kernel Density Estimation
    kde = gaussian_kde(np.vstack([x, y]))
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xgrid, ygrid = np.meshgrid(np.linspace(xmin, xmax, grid_size), np.linspace(ymin, ymax, grid_size))
    z = kde(np.vstack([xgrid.flatten(), ygrid.flatten()]))
    z = np.reshape(z, xgrid.shape)

    # print the values, so see what threshold to use with scientific notation and 3 decimals
    print(f"Density: Min: {z.min():.3e} | Median: {np.median(z):.3e} | Mean: {z.mean():.3e} | Max: {z.max():.3e}")

    # Determine density threshold based on input
    if density_threshold == 'median':
        density_threshold = np.median(z)
    elif density_threshold == 'mean':
        density_threshold = z.mean()
    elif isinstance(density_threshold, (float, int)):
        pass
    else:
        raise ValueError("Invalid density_threshold. Choose 'median', 'mean', or a specific numeric value.")

    # Plot the KDE and contours with specified levels
    fig, ax = plt.subplots(figsize=(5, 5)) if plot_results else plt.subplots()
    cset = ax.contour(xgrid, ygrid, z, levels=[density_threshold], cmap='Reds')
    plt.close(fig)  # Close the plot if not needed

    # Extract polygons for high-density areas
    contour_paths = []
    for i, collection in enumerate(cset.allsegs):
        for seg in collection:
            if len(seg) >= 3:  # Check if the segment can form a valid polygon
                polygon = Polygon(seg)
                if polygon.is_valid:
                    contour_paths.append(polygon)

    if not contour_paths:  # Check if no valid polygons were formed
        raise ValueError("No valid high-density areas found with the specified threshold.")

    # Union of all high-density area polygons into a single geometry and simplify
    unioned_polygons = unary_union(contour_paths)
    simplified_polygon = unioned_polygons.simplify(simplification_tolerance, preserve_topology=True)
    if not simplified_polygon.is_valid:
        simplified_polygon = simplified_polygon.buffer(100)

    # Convert the resulting polygon into a GeoDataFrame
    final_polygon_gdf = gpd.GeoDataFrame([{'geometry': simplified_polygon}], crs=event_points_gdf.crs)

    # Optionally plot the final result
    if plot_results:
        fig, ax = plt.subplots(figsize=(10, 10))
        final_polygon_gdf.boundary.plot(ax=ax, color='blue')
        event_points_gdf.plot(ax=ax, color='red', markersize=5)
        plt.title('Final Area of Interest')
        plt.show()
    return final_polygon_gdf


# Method to plot the population density, simulated event points, and high population density areas side-by-side
def plot_population_density_and_event_points(district_boundary, population_gdf, event_points_gdf, high_pop_density_area, edges, figsize=(10, 10)):
    # Adjusted figure size and added gridspec_kw for spacing
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'wspace': 0.5})
    vmin, vmax = 0, 200

    # Population Density
    district_boundary.boundary.plot(ax=ax1, color='black', linewidth=0.5, alpha=0.7)
    population_gdf.plot(ax=ax1, column='population', cmap='Reds', legend=True, alpha=0.7, vmin=vmin, vmax=vmax)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Population density', pad=20)

    # Simulated Event Points - based on population density
    district_boundary.boundary.plot(ax=ax2, color='black', linewidth=0.5, alpha=0.7)
    event_points_gdf.plot(ax=ax2, color='red', markersize=10, alpha=0.7, edgecolor='black', lw=0.5)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title(f"Simulated event points: {len(event_points_gdf)}", pad=20)

    # Combined plot of Simulated Event Points and (derived) High Population Density Areas
    edges.plot(ax=ax3, linewidth=0.3, color='gray', alpha=0.5, zorder=-1)
    district_boundary.boundary.plot(ax=ax3, color='black', linewidth=0.5, alpha=0.7)
    high_pop_density_area.boundary.plot(ax=ax3, color='blue', linewidth=2, alpha=0.7)
    event_points_gdf.plot(ax=ax3, color='red', markersize=10, alpha=0.7, edgecolor='black', lw=0.5)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title('High population density areas', pad=20)
    plt.show()


#######################################################################
# CAR POINTS
#######################################################################

# Function to filter and plot nodes by centrality (closeness or betweenness)
def filter_by_centrality(geo_df, district_boundary, top_percent, bottom_percent, input_graph, centrality_measure, plot=False):
    """
    Filters a GeoDataFrame to find the top X% and bottom Y% of nodes based on closeness_centrality or betweenness_centrality,
    optionally plots these nodes, and returns a filtered GeoDataFrame excluding the bottom Y% nodes.
    
    :param geo_df: GeoDataFrame with columns for 'closeness_centrality' and 'betweenness_centrality' column and 'x', 'y' for plotting.
    :param top_percent: The top percentage of nodes to select based on chosen centrality measure.
    :param bottom_percent: The bottom percentage of nodes to select based on chosen centrality measure.
    :param plot: Whether to plot the nodes.
    :return: A filtered GeoDataFrame excluding the Y% nodes with lowest centrality score.
    """
    # Calculate the number of nodes for each selection
    num_top = int(len(geo_df) * top_percent)
    num_bottom = int(len(geo_df) * bottom_percent)
    
    # Sort the DataFrame by centrality
    if centrality_measure == 'closeness':
        sorted_geo_df = geo_df.sort_values(by='closeness_centrality', ascending=False)
    elif centrality_measure == 'betweenness':
        sorted_geo_df = geo_df.sort_values(by='betweenness_centrality', ascending=False)
    else:
        raise ValueError(f"Centrality measure '{centrality_measure}' not recognized. Use 'closeness' or 'betweenness'.")
    
    # Select the top X% and bottom Y%
    central_car_nodes = sorted_geo_df.head(num_top)
    remote_car_nodes = sorted_geo_df.tail(num_bottom)

    print(f"Input nr of car nodes: {len(geo_df)}")
    print(f"Remaining nr of car nodes: {len(sorted_geo_df) - len(remote_car_nodes)}, after discarding the {len(remote_car_nodes)} ({bottom_percent*100:.0f}%) remote car nodes with lowest {centrality_measure} centrality")
    print("Centrality Measure:", centrality_measure)
    if plot:
        # Plot all nodes
        fig, ax = ox.plot_graph(input_graph, node_color="white", node_size=0, edge_linewidth=0.2, edge_color="w", show=False, close=False)
        district_boundary.boundary.plot(ax=ax, color='green', linewidth=2.5, alpha=0.7)
        ax.scatter(geo_df['x'], geo_df['y'], c='white', s=50, label="Input Car nodes")
        ax.scatter(central_car_nodes['x'], central_car_nodes['y'], c='orange', s=50, label=f"Highest {top_percent*100:.0f}% {centrality_measure} centrality")
        ax.scatter(remote_car_nodes['x'], remote_car_nodes['y'], c='red', s=50, label=f"Lowest {bottom_percent*100:.0f}% {centrality_measure} centrality")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
        plt.show()
    
    # Remove the bottom Y% nodes from the original GeoDataFrame
    filtered_geo_df = sorted_geo_df.drop(remote_car_nodes.index)
    return filtered_geo_df


# Function to filter and plot nodes by proximity to each other (minimum distance)
def filter_nodes_by_proximity(geo_df, district_boundary,  min_distance, input_graph, criterion_col=None, prefer='higher', plot=False):
    """
    Removes nodes from a GeoDataFrame that are within a specified minimum distance of each other.
    
    :param geo_df: GeoDataFrame with a 'geometry' column.
    :param min_distance: Minimum distance in the GeoDataFrame's coordinate reference system units.
    :param input_graph: The graph from which the nodes were extracted, used for plotting.
    :param criterion_col: Column name to use as a criterion for removing nodes. Optional.
    :param prefer: Determines which node to keep based on the criterion_col ('higher' or 'lower').
    :param plot: Whether to plot the nodes.
    :return: A filtered GeoDataFrame excluding nodes within the minimum distance of each other.
    """
    sindex = geo_df.sindex # Create a spatial index for the GeoDataFrame
    to_remove = [] # List to keep track of indices to remove
    
    # Iterate over the GeoDataFrame
    for index, row in geo_df.iterrows():
        # Create a buffer around the geometry and find potential matches using the spatial index
        buffer = row.geometry.buffer(min_distance)
        possible_matches_index = list(sindex.intersection(buffer.bounds))
        possible_matches = geo_df.iloc[possible_matches_index]
        
        # Actual neighbors are those within the specified distance, excluding the row itself
        actual_neighbors = possible_matches[possible_matches.distance(row.geometry) < min_distance]
        actual_neighbors = actual_neighbors.drop(index, errors='ignore')
        
        for neighbor_index, neighbor in actual_neighbors.iterrows():
            if criterion_col:
                # Decide which node to remove based on the criterion
                if (prefer == 'higher' and neighbor[criterion_col] > row[criterion_col]) or \
                   (prefer == 'lower' and neighbor[criterion_col] < row[criterion_col]):
                    to_remove.append(index)
                    break  # Current node will be removed, no need to check other neighbors
                else:
                    to_remove.append(neighbor_index)
            else:
                # If no criterion is given, default to removing the neighbor
                to_remove.append(neighbor_index)
    
    # Remove duplicates and drop the nodes
    to_remove = list(set(to_remove))
    filtered_geo_df = geo_df.drop(index=to_remove)
    
    # Reset index to clean up the DataFrame
    filtered_geo_df.reset_index(drop=True, inplace=True)

    print(f"Input nr of car nodes: {len(geo_df)}")
    print(f"Remaining nr of car nodes: {len(geo_df) - len(to_remove)}, after removing the {len(to_remove)} nodes that are within {min_distance} m of each other\n")

    if plot:
        # Plot all nodes
        fig, ax = ox.plot_graph(input_graph, node_color="white", node_size=0, edge_linewidth=0.2, edge_color="w", show=False, close=False)
        district_boundary.boundary.plot(ax=ax, color='green', linewidth=2.5, alpha=0.7)
        ax.scatter(geo_df.loc[to_remove, 'x'], geo_df.loc[to_remove, 'y'], c='red', s=50, label="Removed car nodes")
        ax.scatter(filtered_geo_df['x'], filtered_geo_df['y'], c='orange', s=50, label=f"Remaining car nodes")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
        plt.show()
    return filtered_geo_df


#######################################################################
# COST MATRIX
#######################################################################

# Function to add suffix to duplicate eventNodeIDs
def suffix_duplicate_nodes(input_df):
    """
    Add suffixes to duplicate eventNodeIDs based on their occurrence count within each carNodeID.

    Parameters:
    - input_df (pandas.DataFrame): The input DataFrame containing the eventNodeID and carNodeID columns.

    Returns:
    - pandas.DataFrame: A new DataFrame with suffixes added to duplicate eventNodeIDs.
    """
    df = input_df.copy()
    df['eventNodeID'] = df['eventNodeID'].astype(str)
    
    # Generate suffixes based on their occurrence count within each carNodeID
    df['suffix'] = df.groupby(['carNodeID', 'eventNodeID']).cumcount().add(1).astype(str)
    
    # Add suffixes to duplicate eventNodeIDs
    df['eventNodeID'] = df['eventNodeID'] + np.where(df['suffix'] != '0', '_' + df['suffix'], '')
    
    # Drop the temporary suffix column
    df = df.drop(columns=['suffix'])
    return df


# Function to convert CostMatrix to dict + problem size reduction
def preprocess_cost_matrix(CostMatrix, discard_threshold=0.30, verbose=True):
    """
    Preprocesses the Cost Matrix by removing duplicates, converting it to a dictionary for faster lookup,
    and reducing the problem size by discarding the highest travel times based on a specified threshold.

    Parameters:
    - CostMatrix (pd.DataFrame): The original cost matrix with columns ['carNodeID', 'eventNodeID', 'travel_time'].
    - discard_threshold (float): The fraction of the highest travel times to discard. Default is 0.30.
    - verbose (bool): If True, prints statistics about the preprocessing. Default is True.

    Returns:
    - dict: A dictionary of the reduced cost matrix with (carNodeID, eventNodeID) as keys and 'travel_time' as values.
    """

    # Remove duplicates while keeping the first occurrence
    CostMatrix_unique = CostMatrix.drop_duplicates(subset=['carNodeID', 'eventNodeID'], keep='first')

    # Convert to a dictionary for fast lookup
    CostMatrix_dict = CostMatrix_unique.set_index(['carNodeID', 'eventNodeID'])['travel_time'].to_dict()

    # Determine the maximum acceptable travel time based on the discard threshold
    max_acceptable_travel_time = np.percentile(list(CostMatrix_dict.values()), 100 * (1 - discard_threshold))

    # Filter out pairs exceeding the maximum acceptable travel time
    CostMatrix_dict_reduced = {pair: time for pair, time in CostMatrix_dict.items() if time <= max_acceptable_travel_time}

    if verbose:
        # Print statistics for the reduced CostMatrix_dict
        print(f"Filtering out {discard_threshold*100:.0f}% highest travel times - keeping only travel times <= {max_acceptable_travel_time:.0f} sec, or {max_acceptable_travel_time/60:.1f} min")
        print(f"Original nr of pairs: {len(CostMatrix_dict)} | Filtered nr of pairs: {len(CostMatrix_dict_reduced)}")
        print(f"Original max travel time: {np.max(list(CostMatrix_dict.values()))} | Filtered max travel time: {np.max(list(CostMatrix_dict_reduced.values()))}")
    return CostMatrix_dict_reduced


#######################################################################
# PuLP
#######################################################################

# Functioon to define PuLP problem
def define_pulp_problem(CostMatrix, CostMatrix_dict_reduced, nr_of_cars=4, car_capacity=300, problem_name="PoliceCarLocationOptimization", verbose=True):
    """
    Defines the PuLP problem for the Police Car Location Optimization.

    Parameters:
    - CostMatrix (pd.DataFrame): The original cost matrix with columns ['carNodeID', 'eventNodeID', 'travel_time'].
    - CostMatrix_dict_reduced (dict): A dictionary of the reduced cost matrix with (carNodeID, eventNodeID) as keys and 'travel_time' as values.
    - nr_of_cars (int): The number of police cars to place in the final solution. Default is 4.
    - car_capacity (int): The maximum number of events a single police car can respond to. Default is 300.
    - problem_name (str): The name of the PuLP problem. Default is "PoliceCarLocationOptimization".
    - verbose (bool): If True, prints statistics about the problem setup. Default is True.

    Returns:
    - pulp.LpProblem: The defined PuLP problem.
    """
    # Sets
    P = CostMatrix['carNodeID'].unique()  # Potential police car locations
    E = CostMatrix['eventNodeID'].unique()  # Events

    # Create the LP object - minimize total travel time
    problem = pulp.LpProblem(problem_name, pulp.LpMinimize) # Minimization problem

    if verbose:
        print(f"Number of police car locations: {len(P)}")
        print(f"Number of events: {len(E)}")

    # Decision Variables
    # x[i] = 1 if a police car is placed at location i, 0 otherwise
    x = pulp.LpVariable.dicts("x", P, cat='Binary')  # Police car placement

    # # y[i, j] = 1 if event j is assigned to police car i, 0 otherwise
    y = pulp.LpVariable.dicts("y", CostMatrix_dict_reduced.keys(), cat='Binary')  # Event assignment

    # Objective Function - Modified to use CostMatrix_dict_reduced for fast lookup
    problem += pulp.lpSum([CostMatrix_dict_reduced[(i, j)] * y[(i, j)] for i in P for j in E if (i, j) in CostMatrix_dict_reduced]), "TotalResponseTime"

    # Constraints
    # Police Car Placement Constraint
    problem += pulp.lpSum([x[i] for i in P]) == nr_of_cars, "NumberOfPoliceCars"

    # Event Assignment Constraint
    for j in E:
        problem += pulp.lpSum([y[(i, j)] for i in P if (i, j) in CostMatrix_dict_reduced]) == 1, f"EventAssignment_{j}"

    # Validity Constraint
    for (i, j) in CostMatrix_dict_reduced:
        problem += y[(i, j)] <= x[i], f"Validity_{i}_{j}"

    # Capacity Constraint
    for i in P:
        problem += pulp.lpSum([y[(i, j)] for j in E if (i, j) in CostMatrix_dict_reduced]) <= car_capacity * x[i], f"Capacity_{i}"

    if verbose:
        # Print statistics about the problem
        print(f"Number of decision variables: {len(problem.variables())}")
        print(f"Number of constraints: {len(problem.constraints)}")
        print(f"Number of non-zero coefficients: {len(problem.variables())}")
        print(f"Number of non-zero coefficients in the objective function: {len(problem.objective)}")
    return problem


# Function to run solvers - first fast LP relaxation, then MILP configuration
def run_solvers(problem, P, nr_of_locations, solver_name='PULP_CBC_CMD', forceMIP=False, plot=False):
    """
    Run the PULP solver with different configurations to find the optimal solution.
    If the faster solver does not find the optimal solution, switch to the slower solver.
    """
    # Use the faster solver first with integer variable relaxation
    if solver_name == 'PULP_CBC_CMD':
        status = problem.solve(pulp.PULP_CBC_CMD(mip=False, msg=False))
    elif solver_name == 'COIN_CMD':
        status = problem.solve(pulp.COIN_CMD(mip=False, msg=False))
    elif solver_name == 'GLPK_CMD':
        status = problem.solve(pulp.GLPK_CMD(mip=False, msg=False, timeLimit=120)) # 2 minutes time limit
    elif solver_name == 'HiGHS':
        status = problem.solve(pulp.HiGHS(mip=False, msg=False, parallel="on"))

    optimal_locations = np.array([i for i in P if problem.variablesDict()[f"x_{i}"].varValue == 1])
    log_df = pd.DataFrame([{'Variable': v.name, 'Value': v.varValue} for v in problem.variables()]) # Log optimization results

    # Verify if nr of locations is correct
    if (len(optimal_locations) == nr_of_locations) and (plot == True):
            # if the fast LP solver finds the correct amount of locations, plot only this one
            print(f"{solver_name} with LP relaxation successful: {optimal_locations}")
            plt.figure(figsize=(5, 3))
            log_df['Value'].sort_values().reset_index(drop=True).plot()
            plt.ylabel('Value'); plt.xlabel('Combination'); plt.title('Optimization Results (Fast LP solver)')
            plt.grid(linestyle='-', alpha=0.5); plt.tight_layout(); plt.show()
    # If solution from fast solver is invalid, switch to slow MILP solver
    if (len(optimal_locations) < nr_of_locations) or forceMIP:
        print(f"{solver_name} with LP relaxation found {len(optimal_locations)}/{nr_of_locations} locations in {problem.solutionTime:.2f} seconds.")
        print("Switching to MILP solver configuration to find optimal solution.\n")
        if solver_name == 'PULP_CBC_CMD':
            status = problem.solve(pulp.PULP_CBC_CMD(mip=True, msg=False))
        elif solver_name == 'COIN_CMD':
            status = problem.solve(pulp.COIN_CMD(mip=True, msg=False))
        elif solver_name == 'GLPK_CMD':
            status = problem.solve(pulp.GLPK_CMD(mip=True, msg=False, timeLimit=120)) # 2 minutes time limit
        elif solver_name == 'HiGHS':
            status = problem.solve(pulp.HiGHS(mip=True, msg=False, parallel="on"))
        optimal_locations = np.array([i for i in P if problem.variablesDict()[f"x_{i}"].varValue == 1])

        if plot:
            log_df2 = pd.DataFrame([{'Variable': v.name, 'Value': v.varValue} for v in problem.variables()]) # Log optimization result
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            log_df['Value'].sort_values().reset_index(drop=True).plot(ax=ax1)
            log_df2['Value'].sort_values().reset_index(drop=True).plot(ax=ax2)
            ax1.set_ylabel('Value'); ax1.set_xlabel('Combinations'); ax1.set_title('Optimization Results (Fast LP solver)')
            ax2.set_ylabel('Value'); ax2.set_xlabel('Combinations'); ax2.set_title('Optimization Results (Slow MILP solver)')
            ax1.grid(linestyle='-', alpha=0.5); ax2.grid(linestyle='-', alpha=0.5); plt.tight_layout(); plt.show()
        
    # Print the stats from final solver
    print(f"Optimal police car locations found: {len(optimal_locations)}/{nr_of_locations} in {problem.solutionTime:.2f} seconds: {optimal_locations}")
    print(f"Solver: {solver_name} | Status: {problem.status} ({pulp.LpStatus[problem.status]})")
    print(f"Objective function value (total response time): {pulp.value(problem.objective):.4f} seconds, or {pulp.value(problem.objective)/60:.2f} minutes, or {pulp.value(problem.objective)/3600:.2f} hours")
    return optimal_locations


# function to create car to events assignment
def create_car_to_events_df(CostMatrix_extended, optimal_locations, problem, car_capacity, nr_of_unique_events, verbose=False):
    """
    Create a Dataframe with the assigned events to each police car based on the PuLP problem solution.
    """
    # Initialize a dictionary to hold the assignment of events to each car
    car_to_events_assignment = {car: [] for car in optimal_locations.keys()}

    # Iterate over the 'y' variables to extract assignments
    for var in problem.variables():
        if var.name.startswith("y") and var.varValue == 1:
            # Extract carNodeID and eventNodeID from the variable name
            _, car_event_pair = var.name.split("_", 1)
            carNodeID, eventNodeID = car_event_pair.strip("()").split(",_")
            carNodeID = int(carNodeID)
            # eventNodeID = int(eventNodeID)
            car_to_events_assignment[carNodeID].append(eventNodeID)
    # Verify assigned events respects the max capacity constraint
    if verbose:
        total_events = 0
        for car, events in car_to_events_assignment.items():
            print(f"Car {car} is assigned {len(events)}/{car_capacity} events")
            total_events += len(events)
        print(f"Summing the events for each car gives {total_events} events, which should equal the total number of unique events: {nr_of_unique_events}")
    
    # Prepare the data for DataFrame creation
    data_for_df = []
    for car, events in car_to_events_assignment.items():
        for event in events:
            data_for_df.append({"carNodeID": car, "eventNodeID": event})

    # remove eventNodeID suffixes ('_1', '_2', etc) and convert column to int64
    for i in range(len(data_for_df)):
        # data_for_df[i]['eventNodeID'] = int(data_for_df[i]['eventNodeID'].split('_')[0])
        data_for_df[i]['eventNodeID'] = int(data_for_df[i]['eventNodeID'].replace("'", "").split('_')[0])

    # Create and merge DataFrame
    car_to_events_df = pd.DataFrame(data_for_df)
    # car_to_events_df = pd.merge(car_to_events_df, CostMatrix_extended[['carNodeID', 'eventNodeID', 'distance', 'travel_time', 'x', 'y']], on=['carNodeID', 'eventNodeID'], how='left')
    CostMatrix_extended = CostMatrix_extended.drop_duplicates(subset=['carNodeID', 'eventNodeID'])
    car_to_events_df = pd.merge(car_to_events_df, CostMatrix_extended[['carNodeID', 'eventNodeID', 'distance', 'travel_time', 'x', 'y']], on=['carNodeID', 'eventNodeID'], how='left')

    # drop duplicates and keep first occurrence
    # car_to_events_df = car_to_events_df.drop_duplicates(subset=['carNodeID', 'eventNodeID'], keep='first')
    return car_to_events_df


#######################################################################
# VISUALIZATION - OPTIMAL LOCATIONS AND EVENT ASSIGNMENTS
#######################################################################

# Method to plot final locations and assigned events
def plot_optimal_allocations(road_network, district_boundary, optimal_locations_gdf, car_to_events_df, 
                car_nodes_gdf_filtered, nr_of_unique_events, nr_of_cars, car_capacity, problem, figsize=(10,10)):

    # plot the optimal police car locations and the events assigned to them
    fig, ax = ox.plot_graph(road_network, node_color="white", node_size=0, bgcolor='k', edge_linewidth=0.2, edge_color="w", show=False, close=False, figsize=figsize)

    # plot Original District Boundary
    district_boundary.boundary.plot(ax=ax, color='green', linewidth=2.5, alpha=0.7)

    # derived unique car locations from optimal_locations_gdf
    carNodeID_list = list(optimal_locations_gdf['carNodeID'])

    # Plotting optimal police car locations
    for i, police_car in enumerate(carNodeID_list):
        ax.scatter(optimal_locations_gdf.loc[i, 'geometry'].x, optimal_locations_gdf.loc[i, 'geometry'].y, c=f'C{i}', 
                marker='*', edgecolor='cyan', linewidth=1.8, s=800, label=f"Police car id: {police_car}", zorder=3)

    # Plotting events assigned to each optimal police car
    for car_id in carNodeID_list:
        assigned_events = car_to_events_df[car_to_events_df['carNodeID'] == car_id]
        event_coords = list(zip(assigned_events['y'], assigned_events['x']))
        ax.scatter([x for _, x in event_coords], [y for y, _ in event_coords], s=75, edgecolor='black', lw=0.80, label=f'Events for car: {car_id}', zorder=2)

    # legend label handling
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.0, 1))

    print("Input parameters:")
    print(f"- Possible police car locations: {len(car_nodes_gdf_filtered)} | Optimal locations in solution: {nr_of_cars}")
    print(f"- Events: {nr_of_unique_events} | Max event capacity per police car: {car_capacity}\n")
    print("Solution from Linear Programming (LP) model:")
    print(f"Goal: minimize objective function (total response time)")
    print(f"Objective function value: {pulp.value(problem.objective):.0f} sec | {pulp.value(problem.objective)/60:.1f} min | {pulp.value(problem.objective)/3600:.2f} hours\n")

    for car_id in car_to_events_df['carNodeID'].unique():
        assigned_events = car_to_events_df[car_to_events_df['carNodeID'] == car_id]
        total_events = len(assigned_events)
        total_response_time = assigned_events['travel_time'].sum() / 60 # convert to minutes
        avg_response_time = assigned_events['travel_time'].mean()  / 60 # convert to minutes
        median_response_time = assigned_events['travel_time'].median() / 60 # convert to minutes
        capacity_usage = (total_events / car_capacity) * 100
        
        print(f"Car id: {car_id} handles {total_events} events | Capacity: {capacity_usage:.2f}% | Total response time: {total_response_time:.2f} min | Median: {median_response_time:.2f} min | Avg: {avg_response_time:.2f} min")
    plt.show()


# Create isochrone polygons
def make_iso_polys(G, trip_times, center_nodes, edge_buff=30, node_buff=0, infill=True):
    """
    Generate isochrone polygons for given center nodes in a graph.
    """
    all_isochrone_polys = []
    for center_node in center_nodes:
        isochrone_polys = []
        for trip_time in sorted(trip_times, reverse=True):
            subgraph = nx.ego_graph(G, center_node, radius=trip_time*60, distance='travel_time')
            if subgraph.number_of_nodes() == 0 or subgraph.number_of_edges() == 0:
                # Skip if subgraph is empty
                continue
            
            # Mapping node IDs to Points
            node_id_to_point = {node: Point(data['x'], data['y']) for node, data in subgraph.nodes(data=True)}
            
            edge_lines = []
            for n_from, n_to in subgraph.edges():
                f_point = node_id_to_point[n_from]
                t_point = node_id_to_point[n_to]
                edge_lines.append(LineString([f_point, t_point]))

            # Buffer nodes and edges, then combine
            n = gpd.GeoSeries([node_id_to_point[node] for node in subgraph.nodes()]).buffer(node_buff)
            e = gpd.GeoSeries(edge_lines).buffer(edge_buff)
            all_gs = list(n) + list(e)
            new_iso = gpd.GeoSeries(all_gs).unary_union
            
            # Handling infill
            if infill and isinstance(new_iso, Polygon):
                new_iso = Polygon(new_iso.exterior)
                
            isochrone_polys.append(new_iso)
        all_isochrone_polys.append(isochrone_polys)
    return all_isochrone_polys
    

# Merge isochrones to prevent overlap
def merge_isochrones(G, isochrone_polys):
    """
    Merge isochrones to prevent overlap.
    """
    # Initialize containers for merged isochrones by range
    merged_short = []; merged_middle = []; merged_long = []

    # Populate the lists with polygons to merge
    for location_polygons in isochrone_polys:
        if len(location_polygons) >= 1:
            merged_short.append(location_polygons[0])  # Assuming first is short-range
        if len(location_polygons) >= 2:
            merged_middle.append(location_polygons[1])  # Assuming second is middle-range
        if len(location_polygons) == 3:
            merged_long.append(location_polygons[2])  # Assuming third is long-range

    # Perform the merging
    merged_short_union = unary_union(merged_short)
    merged_middle_union = unary_union(merged_middle)
    merged_long_union = unary_union(merged_long)
    return merged_short_union, merged_middle_union, merged_long_union


# Function to plot the isochrones on a Leaflet map using osmnx and folium
def plot_leaflet_map(road_network, trip_times, merged_isochrones, background_polygon_gdf, background_poly=False):
    """
    Function to plot the merged isochrones on a Leaflet map using osmnx and folium.

    Parameters:
    - road_network (networkx.MultiDiGraph): The road network graph.
    - trip_times (list): A list of trip times in seconds.
    - merged_isochrones (tuple): A tuple of merged isochrones for short, middle, and long ranges.
    - background_polygon_gdf (geopandas.GeoDataFrame): A GeoDataFrame with the background polygon.
    - background_poly (bool): Whether to include the background polygon. Default is False.

    Returns:
    - folium.Map: The Leaflet map with the isochrones.
    """
    # reverse order of trip_times
    trip_times.sort(reverse=True)

    # Prepare data for GeoDataFrame
    data = {
        'trip_time': trip_times,
        'geometry': [merged_isochrones[0], merged_isochrones[1], merged_isochrones[2]]
    }
    # Convert dictionary to GeoDataFrame
    crs_proj = ox.graph_to_gdfs(road_network, nodes=False).crs  # Adjusted to explicitly state nodes=False
    isochrones_gdf = gpd.GeoDataFrame(data, crs=crs_proj)

    # Visualize the merged isochrones on a Leaflet map
    leaflet_map = isochrones_gdf.explore(
        column='trip_time',  # trip_time column to differentiate the isochrones
        cmap='RdPu',  # color map
        tiles='OpenStreetMap',  # 'OpenStreetMap' for light tiles, 'CartoDB dark_matter' for dark tiles
        style_kwds={'fillOpacity': 0.35, 'lineOpacity': 0.7},  # Adjust opacities as needed
        legend=True,  # Legend to differentiate ranges
        tooltip=True  # Display trip times on hover
    )

    if background_poly:
        # Add background polygon to outline police district, but loose hover-legend functionality
        folium.GeoJson(
            background_polygon_gdf.geometry,
            style_function=lambda x: {'color': 'black', 'weight': 0.5, 'fillOpacity': 0.1, 'lineOpacity': 0.7}
        ).add_to(leaflet_map)
    return leaflet_map


#######################################################################
# STATISTICS 1 - TABLES
#######################################################################

# Function to compute district-wide travel time statistics
def compute_district_stats(car_to_events_df):
    """Compute district wide travel time statistics in minutes"""
    # Set pandas number of decimal places to 2
    pd.options.display.float_format = '{:.2f}'.format
    
    # compute district wide travel time statistics from car_to_events_df: mean, median, max, min, std, sum, count, etc.
    car_to_events_df["travel_time"].describe()
    min = car_to_events_df["travel_time"].min() / 60 # convert to minutes
    max = car_to_events_df["travel_time"].max() / 60 # convert to minutes
    mean = car_to_events_df["travel_time"].mean() / 60 # convert to minutes
    median = car_to_events_df["travel_time"].median() / 60 # convert to minutes
    std = car_to_events_df["travel_time"].std() / 60 # convert to minutes
    # var = car_to_events_df["travel_time"].var() / 60 # convert to minutes
    sum = car_to_events_df["travel_time"].sum() / 60 # convert to minutes
    count = car_to_events_df["travel_time"].count()
    percentiles = car_to_events_df["travel_time"].quantile([0.20, 0.5, 0.80])  / 60 # convert to minutes

    # Collect in dataframe in order: min, median, mean, max, std, var, sum, count, percentils
    district_stats = pd.DataFrame({"min": min, "median": median, "mean": mean, "max": max, "std": std, "sum": sum, "count": count, "percentile_20": percentiles[0.20], "percentile_50": percentiles[0.5], "percentile_80": percentiles[0.80]}, index=[0])
    return district_stats


# Function to compute within-district travel time statistics (individual cars)
def compute_within_district_stats(car_to_events_df, car_capacity):
    """Compute within district travel time statistics in minutes for each car """
    # Set pandas number of decimal places to 2
    pd.options.display.float_format = '{:.2f}'.format

    # create a dataframe to hold the statistics for each car
    car_stats_list = []
    # iterate over each car and compute the statistics
    for car in car_to_events_df['carNodeID'].unique():
        car_df = car_to_events_df[car_to_events_df['carNodeID'] == car]
        min = car_df["travel_time"].min() / 60 # convert to minutes
        max = car_df["travel_time"].max() / 60 # convert to minutes
        mean = car_df["travel_time"].mean() / 60 # convert to minutes
        median = car_df["travel_time"].median() / 60 # convert to minutes
        std = car_df["travel_time"].std() / 60 # convert to minutes
        # var = car_df["travel_time"].var() / 60 # convert to minutes
        sum = car_df["travel_time"].sum() / 60 # convert to minutes
        count = car_df["travel_time"].count()
        capacity = (count / car_capacity) * 100
        percentiles = car_df["travel_time"].quantile([0.20, 0.5, 0.80])  / 60 # convert to minutes
        car_stats = pd.DataFrame({"min": min, "median": median, "mean": mean, "max": max, "std": std, "sum": sum, "count": count, f"capacity_{car_capacity}": capacity, "percentile_20": percentiles[0.20], "percentile_50": percentiles[0.5], "percentile_80": percentiles[0.80]}, index=[0])
        car_stats['carNodeID'] = car
        car_stats_list.append(car_stats)

    # concatenate the list of dataframes to a single dataframe
    car_stats_df = pd.concat(car_stats_list)
    car_stats_df = car_stats_df[['carNodeID', 'min', 'median', 'mean', 'max', 'std', 'sum', 'count', f"capacity_{car_capacity}", 'percentile_20', 'percentile_50', 'percentile_80']]
    return car_stats_df


#######################################################################
# STATISTICS 2 - PLOTS DISTRICT-WIDE
#######################################################################

# Function to plot the histogram of travel times (district-wide)
def plot_travel_time_histogram_district(car_to_events_df, district_stats, figsize=(8, 4)):
    """Plot histogram of travel times for all cars"""
    fig, ax = plt.subplots(figsize=figsize)
    plt.hist(car_to_events_df['travel_time'] / 60, bins=50, edgecolor='black')
    plt.axvline(x=district_stats['mean'].values[0], color='r', linestyle='--', lw=2.5, label='Mean travel time')
    plt.axvline(x=district_stats['median'].values[0], color='y', linestyle='--', lw=2.5, label='Median travel time')
    plt.axvline(x=district_stats['percentile_80'].values[0], color='black', linestyle='--', lw=2.5, label='80th percentile')
    plt.axvline(x=district_stats['percentile_20'].values[0], color='black', linestyle='--', lw=2.5, label='20th percentile')

    plt.title('Histogram of travel times for all cars')
    plt.xlabel('Travel time [min]')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(); plt.show()

# Function to plot the scatterplot of travel times (district-wide)
def plot_travel_time_scatterplot_district(car_to_events_df, district_stats, figsize=(8, 4)):
    """Plot scatterplot of travel times for all cars"""
    fig, ax = plt.subplots(figsize=figsize)
    for car in car_to_events_df['carNodeID'].unique():
        car_df = car_to_events_df[car_to_events_df['carNodeID'] == car]
        plt.scatter(car_df.index, car_df['travel_time'] / 60, edgecolor='black', alpha=0.4, label=f"Car {car}", s=20)

    plt.axhline(y=district_stats['percentile_80'].values[0], color='black', linestyle='--', lw=2.5, label='80th percentile')
    plt.axhline(y=district_stats['mean'].values[0], color='r', linestyle='--', lw=2.5, label='Mean travel time')
    # plt.axhline(y=district_stats['median'].values[0], color='y', linestyle='--', lw=2.5, label='Median travel time')
    plt.axhline(y=district_stats['percentile_20'].values[0], color='black', linestyle='--', lw=2.5, label='20th percentile')

    plt.title('Scatter plot of travel times for all cars')
    plt.xlabel('Event ID')
    plt.ylabel('Travel time [min]')
    plt.legend(title='Car Node ID', loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(); plt.show()

# Function to plot both boxplot and violin plot side by side (district-wide)
def plot_travel_time_box_violin_district(car_to_events_df, district_stats, figsize=(8, 5)):
    """Plot both boxplot and violin plot side by side for whole district"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    # boxplot
    box = ax1.boxplot(car_to_events_df['travel_time']/60, patch_artist=True, widths=0.75)
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')
    ax1.axhline(y=district_stats['mean'].values[0], color='r', linestyle='--', lw=2.0, alpha=0.5, label='Mean travel time')
    ax1.axhline(y=district_stats['percentile_80'].values[0], color='black', linestyle='--', lw=2.0, alpha=0.5, label='80th percentile')
    ax1.axhline(y=district_stats['percentile_20'].values[0], color='black', linestyle='--', lw=2.0, alpha=0.5, label='20th percentile')
    ax1.set_title('Travel time box plot')
    ax1.set_xlabel('District events')
    ax1.set_ylabel('Travel time [min]')
    ax1.grid()
    # violin plot
    violin = ax2.violinplot(car_to_events_df['travel_time']/60, widths=0.75, showmedians=True)
    for patch in violin['bodies']:
        patch.set_facecolor('deepskyblue')
    ax2.axhline(y=district_stats['mean'].values[0], color='r', linestyle='--', lw=2.0, alpha=0.5, label='Mean travel time')
    ax2.axhline(y=district_stats['percentile_80'].values[0], color='black', linestyle='--', lw=2.0, alpha=0.5, label='80th percentile')
    ax2.axhline(y=district_stats['percentile_20'].values[0], color='black', linestyle='--', lw=2.0, alpha=0.5, label='20th percentile')
    ax2.set_title('Travel time violin plot')
    ax2.set_xlabel('District events')
    # ax2.set_ylabel('Travel time [min]')
    ax2.grid()
    plt.tight_layout(); plt.show()


#######################################################################
# STATISTICS 3 - PLOTS WITHIN DISTRICT
#######################################################################

# Function to plot the boxplot of travel times for each car (within district)
def plot_travel_time_boxplot_cars(car_to_events_df, figsize=(8, 4)):
    """Plot boxplot of travel times for each car"""
    fig, ax = plt.subplots(figsize=figsize)
    box = plt.boxplot([car_to_events_df.loc[car_to_events_df['carNodeID']==id, 'travel_time']/60 for id in car_to_events_df['carNodeID'].unique()], patch_artist=True, widths=0.75)
    colors = plt.cm.Pastel1.colors
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
    plt.title('Boxplot of travel times for each car')
    plt.suptitle('')
    plt.xlabel('Car Node ID')
    plt.ylabel('Travel time [min]')
    plt.xticks(range(1, len(car_to_events_df['carNodeID'].unique())+1), car_to_events_df['carNodeID'].unique())
    plt.grid(); plt.show()


# Function to plot the violin plot of travel times for each car
def plot_travel_time_violinplot_cars(car_to_events_df, figsize=(8, 4)):
    """Plot violin plot of travel times for each car"""
    fig, ax = plt.subplots(figsize=figsize)
    violin = plt.violinplot([car_to_events_df.loc[car_to_events_df['carNodeID']==id, 'travel_time']/60 for id in car_to_events_df['carNodeID'].unique()], widths=0.75, showmedians=True)
    colors = list(mcolors.TABLEAU_COLORS.values())
    for i, patch in enumerate(violin['bodies']):
        patch.set_facecolor(colors[i % len(colors)])
    plt.title('Violin plot of travel times for each car')
    plt.xlabel('Car Node ID')
    plt.ylabel('Travel time [min]')
    plt.xticks(range(1, len(car_to_events_df['carNodeID'].unique())+1), car_to_events_df['carNodeID'].unique())
    plt.grid(); plt.show()


# Function to plot boxplot and violin plot side by side
def plot_travel_time_box_violin_cars(car_to_events_df, figsize=(8, 5)):
    """Plot both boxplot and violin plot side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    # boxplot
    box = ax1.boxplot([car_to_events_df.loc[car_to_events_df['carNodeID']==id, 'travel_time']/60 for id in car_to_events_df['carNodeID'].unique()], patch_artist=True, widths=0.75)
    colors = plt.cm.Pastel1.colors
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
    ax1.set_title('Boxplot of travel times for each car')
    ax1.set_xlabel('Car Node ID')
    ax1.set_ylabel('Travel time [min]')
    ax1.set_xticks(range(1, len(car_to_events_df['carNodeID'].unique())+1))
    ax1.set_xticklabels(car_to_events_df['carNodeID'].unique())
    ax1.grid()
    # violin plot
    violin = ax2.violinplot([car_to_events_df.loc[car_to_events_df['carNodeID']==id, 'travel_time']/60 for id in car_to_events_df['carNodeID'].unique()], widths=0.75, showmedians=True)
    colors = list(mcolors.TABLEAU_COLORS.values())
    for i, patch in enumerate(violin['bodies']):
        patch.set_facecolor(colors[i % len(colors)])
    ax2.set_title('Violin plot of travel times for each car')
    ax2.set_xlabel('Car Node ID')
    ax2.set_ylabel('Travel time [min]')
    ax2.set_xticks(range(1, len(car_to_events_df['carNodeID'].unique())+1))
    ax2.set_xticklabels(car_to_events_df['carNodeID'].unique())
    ax2.grid()
    plt.tight_layout(); plt.show()
