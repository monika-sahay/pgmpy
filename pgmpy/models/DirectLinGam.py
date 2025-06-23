# pgmpy/models/direct_lingam.py

from pgmpy.base import DAG

class DirectLinGam(DAG):
    """
    A pgmpy-compatible model class for representing the DAG structure
    learned using the DirectLiNGAM estimator.

    Parameters
    ----------
    edges : list of tuples
        Directed edges in the form (parent, child).
    nodes : list of str, optional
        List of node names.
    """

    def __init__(self, edges=None, nodes=None):
        super().__init__()
        if nodes:
            self.add_nodes_from(nodes)
        if edges:
            self.add_edges_from(edges)

    @classmethod
    def from_adjacency_matrix(cls, matrix, column_names):
        """
        Create a DirectLiNGAMModel from an adjacency matrix.

        Parameters
        ----------
        matrix : 2D numpy array
            Adjacency matrix (non-zero values imply edges).
        column_names : list of str
            Column names corresponding to each node.

        Returns
        -------
        model : DirectLiNGAMModel
            The constructed DAG model.
        """
        edges = []
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val != 0:
                    edges.append((column_names[j], column_names[i]))
        return cls(edges=edges, nodes=column_names)
