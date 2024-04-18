import numpy as np
def select_next_node_v2(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """Select the next node to visit from the unvisited nodes with improved constructive heuristics."""

    threshold = 0.5
    c1, c2, c3, c4 = 0.4, 0.3, 0.2, 0.1
    scores = {}

    for node in unvisited_nodes:
        unvisited_node_list = list(unvisited_nodes - {node})
        
        unvisited_distances = distance_matrix[node][unvisited_node_list]
        unvisited_distances = np.append(unvisited_distances, [distance_matrix[node][current_node], distance_matrix[node][destination_node]])
        average_distance_to_unvisited = np.mean(unvisited_distances)
        std_dev_distance_to_unvisited = np.std(unvisited_distances)

        next_unvisited_distances = distance_matrix[unvisited_node_list][:, unvisited_node_list]
        next_unvisited_distances = np.append(next_unvisited_distances.flatten(), [distance_matrix[unvisited_node_list][:, current_node], distance_matrix[unvisited_node_list][:, destination_node]])
        next_average_distance_to_unvisited = np.mean(next_unvisited_distances)
        next_std_dev_distance_to_unvisited = np.std(next_unvisited_distances)

        next_scores = c1 * distance_matrix[current_node][unvisited_node_list] - c2 * next_average_distance_to_unvisited + c3 * next_std_dev_distance_to_unvisited - c4 * distance_matrix[destination_node][unvisited_node_list]
        lookahead_score = np.mean(next_scores) if len(next_scores) > 0 else 0

        score = c1 * distance_matrix[current_node][node] - c2 * average_distance_to_unvisited + c3 * std_dev_distance_to_unvisited - c4 * distance_matrix[destination_node][node] + threshold * lookahead_score
        scores[node] = score

    next_node = min(scores, key=scores.get)
    return next_node
