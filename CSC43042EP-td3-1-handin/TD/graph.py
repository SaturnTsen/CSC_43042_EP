import numpy as np
from TD.cloud import Point, Cloud
from typing import List

class Edge:
    """An edge in a Point graph.

    Attributes
    ----------
    p1, p2 : int -- the vertices (indices of points) connected by the edge
    length : float -- the length of the edge
    """

    def __init__(self, p1:int, p2:int, length:float):
        self.p1 = p1
        self.p2 = p2
        self.length = length

    def __repr__(self):
        return f"Edge({self.p1}, {self.p2}, {self.length})"

    def __str__(self):
        return f"Edge: {self.p1} -> {self.p2} (length {self.length})"

    # Exercise 1
    def __lt__(self, other):
        return self.length < other.length


class Graph:
    """
    A simple weighted graph, with edges sorted by non-decreasing length.
    node_names are to indices of Points in an associated Cloud object.

    Attributes
    ----------
    edges : List[Edge]
    node_names : List[str]
    """

    def __init__(self):
        self.edges : List[Edge] = [] # Assumend to be sorted by non-decreasing edge length
        self.node_names : List[str] = []

    def __str__(self):
        n = len(self.node_names)
        if n == 0:
            node_str = "0 Nodes"
        elif n == 1:
            node_str = f"1 Node: 0: {self.node_names[0]}"
        else:
            node_str = f"{n} Nodes: " + ", ".join(
                f"{i}: {n}" for (i, n) in enumerate(self.node_names)
            )
        m = len(self.edges)
        if m == 0:
            edge_str = "0 Edges"
        elif m == 1:
            edge_str = f"1 Edge: {self.edges}"
        else:
            edge_str = f"{m} Edges: {self.edges}"
        return f"Graph:\n{node_str}\n{edge_str}"

    def __iter__(self):
        # Iterate over a Graph -> iterate over its edges list
        return iter(self.edges)

    def edge_count(self):
        return len(self.edges)

    def node_count(self):
        return len(self.node_names)

    def get_name(self, i: int) -> str:
        return self.node_names[i]

    def get_edge(self, i: int) -> Edge:
        """The i-th edge of the (length-sorted) edge list."""
        return self.edges[i]

    def add_nodes(self, ns: List[str]) -> None:
        """Add a list of (names of) nodes to the graph."""
        # Exercise 2
        self.node_names.extend(ns)

    def add_edges(self, es: List[Edge]) -> None:
        """Add a list of edges to the graph,
        maintaining the length-sorted invariant.
        """
        # Exercise 3
        self.edges.extend(es)
        self.edges.sort()


def graph_from_cloud(c: Cloud):
    """Construct the complete graph whose nodes are names of points in c
    and where the length of the edge between two points is the Euclidean
    distance between them.
    """
    res = Graph()
    # Exercise 4
    length = len(c)
    res.add_nodes([point.name for point in c])
    res.add_edges([
        Edge(j, i, np.sqrt(np.sum((c[i].coords-c[j].coords)**2)).item())
        for i in range(length)
        for j in range(i+1, length)
    ])
    return res


def graph_from_matrix(node_names: List[str], dist_matrix: List[List[float]]):
    """Construct the complete graph on the given list of node names
    with the length of the edge between nodes i and j given by the
    (i,j)-th entry of the matrix.
    """
    length = len(dist_matrix)
    assert len(dist_matrix) == len(dist_matrix[0]) == len(node_names)
    res = Graph()

    # Exercise 5
    res.add_nodes(node_names)    
    res.add_edges([
        Edge(j, i, dist_matrix[i][j])
        for i in range(length)
        for j in range(i+1, length)
    ])
    return res


def graph_from_matrix_file(filename: str):
    """Construct the graph specified in the named file.  The first line
    in the file is the number n of nodes; the next n lines give the node
    names; and the following n lines are the rows of the distance matrix
    (n entries per line, comma-separated).
    """
    # Exercise 6
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    
    n = int(lines[0].strip())
    
    node_names = [line.strip() for line in lines[1:n+1]]
    data_lines = lines[n+1 : 2*n+1]
    matrix = []
    
    for row in data_lines:
        row_values = [float(col.strip()) for col in row.split(',')]
        matrix.append(row_values)
        
    return graph_from_matrix(node_names, matrix)

    

