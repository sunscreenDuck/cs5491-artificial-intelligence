from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import jit, prange, njit
import errno
import os
import signal
import functools
from scipy.spatial import distance_matrix
import stopit


class CustomTimeoutError(Exception):
    pass


def heuristic_aco(num_nodes, _distance_matrix, pheromone_matrix, _alpha, _beta):
    current_node = np.random.randint(num_nodes)
    visited = np.zeros(num_nodes, dtype=bool)
    visited[current_node] = True
    path = [current_node]
    distance = 0

    for _ in range(num_nodes - 1):
        unvisited_nodes = np.where(~visited)[0]
        pheromone_values = pheromone_matrix[current_node][unvisited_nodes]
        attractiveness = _distance_matrix[current_node][unvisited_nodes] ** _alpha
        desirability = (1.0 / _distance_matrix[current_node][unvisited_nodes]) ** _beta
        probabilities = pheromone_values * attractiveness * desirability
        probabilities /= np.sum(probabilities)

        if np.isnan(probabilities).all() or np.sum(probabilities) == 0:
            next_node = np.random.choice(unvisited_nodes)
        else:
            next_node = np.random.choice(unvisited_nodes, p=probabilities)

        path.append(next_node)
        visited[next_node] = True
        distance += _distance_matrix[current_node][next_node]
        current_node = next_node

    # Return to the starting node
    path.append(path[0])
    distance += _distance_matrix[current_node][path[0]]

    return path, distance


class AcoTsp:
    def __init__(self, distance_matrix: np.ndarray, num_ants, num_iterations, evaporation_rate, alpha, beta,
                 local_search=True):
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.local_search = local_search
        self.num_nodes = distance_matrix.shape[0]
        self.pheromone_matrix = self.initialize_pheromone_matrix()
        self.best_path = None
        self.best_distance = np.inf

    def initialize_pheromone_matrix(self):
        return np.ones((self.num_nodes, self.num_nodes))

    def construct_ant_path(self):
        return heuristic_aco(self.num_nodes, self.distance_matrix, self.pheromone_matrix, self.alpha, self.beta)

    def run(self):
        for _ in range(self.num_iterations):
            ant_paths, ant_distances = self.run_ant_colony()
            self.update_best_path_distance(ant_paths, ant_distances)

            self.pheromone_matrix *= self.evaporation_rate
            self.update_pheromone_matrix(ant_paths, ant_distances)

            if self.local_search:
                self.local_search_2opt()

        return self.best_path, self.best_distance

    def run_ant_colony(self):
        ant_paths = []
        ant_distances = []

        with ThreadPoolExecutor(max_workers=None) as executor:
            futures = [executor.submit(self.construct_ant_path) for _ in range(self.num_ants)]
            for future in futures:
                ant_path, ant_distance = future.result()
                ant_paths.append(ant_path)
                ant_distances.append(ant_distance)

        return ant_paths, ant_distances

    def update_best_path_distance(self, ant_paths, ant_distances):
        min_distance_index = np.argmin(ant_distances)
        if ant_distances[min_distance_index] < self.best_distance:
            self.best_distance = ant_distances[min_distance_index]
            self.best_path = ant_paths[min_distance_index]

    def update_pheromone_matrix(self, ant_paths, ant_distances):
        num_nodes = self.num_nodes

        for ant_path, ant_distance in zip(ant_paths, ant_distances):
            for i in prange(num_nodes - 1):  # Use prange for parallel loop
                self.pheromone_matrix[ant_path[i]][ant_path[i + 1]] += 1.0 / ant_distance

    def local_search_2opt(self):
        improved = True

        while improved:
            improved = False
            for i in range(1, len(self.best_path) - 2):
                for j in range(i + 1, len(self.best_path)):
                    if j - i == 1:
                        continue
                    new_path = self.best_path[:]
                    new_path[i:j] = self.best_path[j - 1:i - 1:-1]
                    new_distance = self.calculate_path_distance(new_path)
                    if new_distance < self.best_distance:
                        self.best_path = new_path
                        self.best_distance = new_distance
                        improved = True

    def calculate_path_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i]][path[i + 1]]
        return distance


if __name__ == "__main__":
    import os
    import tsplib95
    import networkx
    import warnings
    import time

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    print("[*] Running ...")

    basepath = os.path.dirname(__file__)
    tsp_fp = os.path.join(basepath, f"tsp/lin105.tsp")
    problem = tsplib95.load(tsp_fp)
    dist_mat = np.array(networkx.adjacency_matrix(problem.get_graph()).todense())

    num_ants = 10
    num_iterations = 100
    evaporation_rate = 0.5
    alpha = 1
    beta = 2

    timeout = 0.2 * 60  # 5 minutes (in seconds)
    start_time = time.time()

    with stopit.ThreadingTimeout(5) as context_manager:
        aco_tsp = AcoTsp(dist_mat, num_ants, num_iterations, evaporation_rate, alpha, beta)
        best_path, best_distance = aco_tsp.run()
    if context_manager.state == context_manager.EXECUTED:
        print(f"Completed : {best_distance}")
    elif context_manager.state == context_manager.TIMED_OUT:
        print(f"Timeout")
