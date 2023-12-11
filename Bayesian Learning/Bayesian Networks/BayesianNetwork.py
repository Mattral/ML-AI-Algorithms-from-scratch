import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class BayesianNetwork:
    def __init__(self, nodes, edges):
        """
        Initialize a Bayesian Network.

        Parameters:
        - nodes (list): List of node names.
        - edges (list): List of directed edges as tuples.
        """
        self.nodes = nodes
        self.edges = edges
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

    def plot(self):
        """
        Visualize the Bayesian Network.
        """
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", arrowsize=20)
        plt.show()

    def conditional_probability(self, node, evidence):
        """
        Dummy function for conditional probability. Replace with actual implementation.

        Parameters:
        - node (str): The node for which conditional probability is calculated.
        - evidence (dict): A dictionary containing evidence nodes and their values.

        Returns:
        - float: Conditional probability P(node | evidence).
        """
        # Placeholder, replace with actual implementation based on your data and domain.
        return np.random.rand()

    def infer(self, query, evidence):
        """
        Perform Bayesian inference.

        Parameters:
        - query (str): The node for which probability is queried.
        - evidence (dict): A dictionary containing evidence nodes and their values.

        Returns:
        - float: Probability P(query | evidence).
        """
        numerator = 1.0
        denominator = 1.0

        for node in self.nodes:
            if node == query:
                numerator *= self.conditional_probability(node, evidence)
            elif node in evidence:
                denominator *= self.conditional_probability(node, evidence)
            else:
                numerator *= self.conditional_probability(node, evidence)
                denominator *= self.conditional_probability(node, evidence)

        result = numerator / denominator
        return result

# Example Usage
nodes = ["A", "B", "C", "D"]
edges = [("A", "B"), ("B", "C"), ("C", "D")]

bayesian_network = BayesianNetwork(nodes, edges)

# Print information about the network
print("Nodes:", bayesian_network.nodes)
print("Edges:", bayesian_network.edges)

# Visualize the network
print("Visualizing the Bayesian Network:")
bayesian_network.plot()

# Example query
query_node = "D"
evidence_nodes = {"A": 1, "B": 0, "C": 1}

# Perform inference and print the result
result = bayesian_network.infer(query_node, evidence_nodes)
print(f"\nP({query_node} | {evidence_nodes}) = {result:.4f}")
