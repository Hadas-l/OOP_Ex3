import copy


class DiGraph:
    """
    This class represents a bi-directional graph
    with positive weights.

    Graph functionality as follows:
    -- add node
    -- add edge
    -- get number of nodes in the graph
    -- get number of edges in the graph
    -- get number of operations done on the graph
    -- get neighbors of a node
    -- remove node
    -- remove edge
    """

    class DiNode:
        """
        A simple class to represent a Node
        in a graph with simple data : key, position
        """
        def __init__(self, key, pos=(0, 0, 0)):
            self.id = key
            self.pos = pos

            self.ins = 0
            self.outs = 0

        def __repr__(self):
            return "{}: |edges out| {} |edges in| {}".format(self.id, self.outs, self.ins)

        def __eq__(self, other):
            return self.id == other.id and self.pos == other.pos

    def __init__(self, graph=None):
        self.nodes = {}
        self.neighbors = {}
        self.connected_to = {}

        self.mc = 0
        self.edge_size = 0
        self.node_size = 0

        if graph:
            self.nodes = copy.deepcopy(graph.nodes)
            self.neighbors = copy.deepcopy(graph.neighbors)
            self.connected_to = copy.deepcopy(graph.connected_to)

            self.mc = graph.mc
            self.edge_size = graph.edge_size
            self.node_size = graph.node_size

    def v_size(self) -> int:
        """
        Returns the number of vertices in this graph
        @return: The number of vertices in this graph
        """
        return self.node_size

    def e_size(self) -> int:
        """
        Returns the number of edges in this graph
        @return: The number of edges in this graph
        """
        return self.edge_size

    def get_all_v(self) -> dict:
        """return a dictionary of all the nodes in the Graph, each node is represented using apair  (key, node_data)
        """
        return self.nodes

    def all_in_edges_of_node(self, id1: int) -> dict:
        """return a dictionary of all the nodes connected to (into) node_id ,
        each node is represented using a pair (key, weight)
         """

        return self.connected_to[id1] if id1 in self.nodes else {}

    def all_out_edges_of_node(self, id1: int) -> dict:
        """return a dictionary of all the nodes connected from node_id , each node is represented using a pair (key,
        weight)
        """

        return self.neighbors[id1] if id1 in self.nodes else {}

    def get_mc(self) -> int:
        """
        Returns the current version of this graph,
        on every change in the graph state - the MC should be increased
        @return: The current version of this graph.
        """
        return self.mc

    def add_edge(self, id1: int, id2: int, weight: float) -> bool:
        """
        Adds an edge to the graph.
        @param id1: The start node of the edge
        @param id2: The end node of the edge
        @param weight: The weight of the edge
        @return: True if the edge was added successfully, False o.w.
        Note: If the edge already exists or one of the nodes dose not exists the functions will do nothing
        """

        if all(key in self.nodes for key in (id1, id2)):

            if weight >= 0:

                if id2 not in self.neighbors[id1] and id1 not in self.connected_to[id2]:

                    self.neighbors[id1][id2] = weight
                    self.connected_to[id2][id1] = weight

                    self.nodes[id1].outs += 1
                    self.nodes[id2].ins += 1

                    self.edge_size += 1
                    self.mc += 1

                    return True

        return False

    def add_node(self, node_id: int, pos: tuple = None) -> bool:
        """
        Adds a node to the graph.
        @param node_id: The node ID
        @param pos: The position of the node
        @return: True if the node was added successfully, False o.w.
        Note: if the node id already exists the node will not be added
        """

        if node_id not in self.nodes:

            if pos:
                self.nodes[node_id] = DiGraph.DiNode(node_id, pos)
            else:
                self.nodes[node_id] = DiGraph.DiNode(node_id)

            self.neighbors[node_id] = {}
            self.connected_to[node_id] = {}

            self.node_size += 1
            self.mc += 1
            return True

        return False

    def remove_node(self, node_id: int) -> bool:
        """
        Removes a node from the graph.
        @param node_id: The node ID
        @return: True if the node was removed successfully, False o.w.
        Note: if the node id does not exists the function will do nothing
        """

        if node_id in self.nodes:

            for n in self.neighbors[node_id]:
                del self.connected_to[n][node_id]
                self.edge_size -= 1
                self.nodes[n].ins -= 1

            for n in self.connected_to[node_id]:
                del self.neighbors[n][node_id]
                self.edge_size -= 1
                self.nodes[n].outs -= 1

            self.node_size -= 1
            self.mc += 1
            del self.nodes[node_id]

            return True

        return False

    def remove_edge(self, node_id1: int, node_id2: int) -> bool:
        """
        Removes an edge from the graph.
        @param node_id1: The start node of the edge
        @param node_id2: The end node of the edge
        @return: True if the edge was removed successfully, False o.w.
        Note: If such an edge does not exists the function will do nothing
        """

        if all(key in self.nodes for key in (node_id1, node_id2)):

            if node_id2 in self.neighbors[node_id1] and node_id1 in self.connected_to[node_id2]:

                del self.neighbors[node_id1][node_id2]
                del self.connected_to[node_id2][node_id1]

                self.nodes[node_id1].outs -= 1
                self.nodes[node_id1].ins -= 1

                self.edge_size -= 1
                self.mc += 1

                return True

        return False

    def transpose(self):
        """
        transpose graph
        :return: transposed graph
        """

        transposed = DiGraph()

        for node in self.get_all_v():
            transposed.add_node(node)

        for node in self.get_all_v():
            for ni in self.all_out_edges_of_node(node):
                transposed.add_edge(ni, node, self.neighbors[node][ni])

        return transposed

    def __repr__(self):
        return "|V|={} , |E|={}".format(self.v_size(), self.e_size())

    def __str__(self):
        return "|V|={} , |E|={}".format(self.v_size(), self.e_size())

    def __eq__(self, other):
        return self.nodes == other.nodes and self.edge_size == other.edge_size \
               and self.node_size == other.node_size and self.connected_to == other.connected_to \
               and self.neighbors == other.neighbors
