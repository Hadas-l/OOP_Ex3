from typing import List
from src import GraphInterface
import json
import sys
from src.DiGraph import DiGraph
import heapq
import matplotlib.pyplot as plt
import random


class GraphAlgo:

    def __init__(self, graph=None):
        self.graph = graph

    def get_graph(self) -> GraphInterface:
        """
        :return: the directed graph on which the algorithm works on.
        """
        return self.graph

    def load_from_json(self, file_name: str) -> bool:
        """
        Loads a graph from a json file.
        @param file_name: The path to the json file
        @returns True if the loading was successful, False o.w.
        """

        try:

            with open(file_name, 'r') as file:

                graph_dict = json.load(file)

                graph = DiGraph()

                for node_json in graph_dict['Nodes']:

                    try:
                        split_pos = node_json['pos'].split(",")
                        x, y, z = float(split_pos[0]), float(split_pos[1]), float(split_pos[2])
                        graph.add_node(node_json['id'], (x, y, z))

                    except:
                        graph.add_node(node_json['id'])

                for edge_json in graph_dict["Edges"]:
                    graph.add_edge(edge_json['src'], edge_json['dest'], edge_json['w'])

                self.graph = DiGraph(graph)
                return True

        except OSError as err:
            print("error: {}".format(err))
            return False

        except:
            print("unexpected error ", sys.exc_info()[0])
            return False

    def save_to_json(self, file_name: str) -> bool:  # {
        """
        Saves the graph in JSON format to a file
        @param file_name: The path to the out file
        @return: True if the save was successful, Flase o.w.
        """
        try:

            graph_json = {"Nodes": [], "Edges": []}

            for node_key in self.graph.get_all_v():
                node = self.graph.get_all_v()[node_key]

                # convert pos from tuple to string
                x, y, z = node.pos
                pos_string = "{},{},{}".format(x, y, z)

                graph_json["Nodes"].append({
                    "id": node_key,
                    "pos": pos_string
                })

                for out in self.graph.all_out_edges_of_node(node_key):

                    graph_json["Edges"].append({
                        "src": node_key,
                        "dest": out,
                        "w": self.graph.all_out_edges_of_node(node_key)[out]
                    })

            with open(file_name, 'w') as file:
                json.dump(graph_json, file)

                return True

        except OSError as err:
            print("error: {}".format(err))
            return False

        except:
            print("unexpected error ", sys.exc_info()[0])
            return False

    def djikstra(self, src_node: int) -> (list, list):
        """
        Djikstra's algorithm with priority queue

        More info:
        https://en.wikipedia.org/wiki/Dijkstra's_algorithm
        :param src_node: starting node
        :return:
        """

        distance = {i: float('inf') for i in self.graph.get_all_v()}
        parents = {i: None for i in self.graph.get_all_v()}

        distance[src_node] = 0.0
        priority_queue = [(distance[src_node], src_node)]

        visited = set()

        while priority_queue:

            _, key = heapq.heappop(priority_queue)

            if key not in visited:
                visited.add(key)

                for ni_key in self.graph.all_out_edges_of_node(key):

                    if ni_key not in visited:
                        # visited.add(ni_key)

                        alt = _ + self.graph.all_out_edges_of_node(key)[ni_key]

                        if alt < distance[ni_key]:
                            distance[ni_key] = alt
                            parents[ni_key] = key

                            heapq.heappush(priority_queue, (alt, ni_key))

        return distance, parents

    def shortest_path(self, id1: int, id2: int) -> (float, list):
        """
        Returns the shortest path from node id1 to node id2 using Dijkstra's Algorithm
        @param id1: The start node id
        @param id2: The end node id
        @return: The distance of the path, the path as a list
        Example:
#      >>> from GraphAlgo import GraphAlgo
#       >>> g_algo = GraphAlgo()
#        >>> g_algo.addNode(0)
#        >>> g_algo.addNode(1)
#        >>> g_algo.addNode(2)
#        >>> g_algo.addEdge(0,1,1)
#        >>> g_algo.addEdge(1,2,4)
#        >>> g_algo.shortestPath(0,1)
#        (1, [0, 1])
#        >>> g_algo.shortestPath(0,2)
#        (5, [0, 1, 2])
        More info:
        https://en.wikipedia.org/wiki/Dijkstra's_algorithm
        """
        if id1 in self.graph.get_all_v() and id2 in self.graph.get_all_v() and id1 is not id2:

            distance, parents = self.djikstra(id1)
            shortest_dist = distance[id2]

            parent_key = id2
            shortest_path = [parent_key]

            while parent_key is not None:

                shortest_path.append(parents[parent_key])
                parent_key = parents[parent_key]

                if parent_key == id1:
                    return shortest_dist, shortest_path[::-1]

                if parent_key is None:
                    return shortest_dist, []

        return float('inf'), []

    def dfs(self, v: int, visited: dict, final, graph=None) -> list:
        """
        utill function to scan graph recursively with DFS
        :param graph: transposed graph - default to none
        :param final: strongly connected component list
        :param nodes: dict of nodes for the current graph dfs will run on
        :param v: current node
        :param visited: list of visited vertexes
        :return: list representing a strongly connected component
        """
        if graph:
            visited[v] = True

            final.append(v)

            for out in graph.all_out_edges_of_node(v):
                if not visited[out]:
                    self.dfs(out, visited, final, graph)

            return final

        else:
            visited[v] = True

            outs = self.graph.all_out_edges_of_node(v)
            for out in outs:
                if not visited[out]:
                    self.dfs(self.graph.get_all_v()[out].id, visited, final)
            final.append(v)

    def connected_component(self, id1: int) -> list:
        """
        Finds the Strongly Connected Component(SCC) that node id1 is a part of.
        @param id1: The node id
        @return: The list of nodes in the SCC
        """

        for i in self.connected_components():
            if id1 in i:
                return i

        return []

    def connected_components(self) -> List[list]:
        """
        Finds all the Strongly Connected Component(SCC) in the graph.
        @return: The list all SCC
        """
        stack = []

        visited = {i: False for i in self.graph.get_all_v()}

        for v in self.graph.get_all_v():
            if not visited[v]:
                self.dfs(v, visited, stack)

        transposed = self.get_graph().transpose()

        visited = {i: False for i in self.graph.get_all_v()}

        res = []

        while stack:
            key = stack.pop()
            if not visited[key]:
                component = self.dfs(key, visited, [], graph=transposed)
                res.append(component)

        return res

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """

        nodes = self.graph.get_all_v()

        for n in nodes:
            x, y, z = random.uniform(0, self.graph.v_size()), random.uniform(0, self.graph.v_size()), 0
            nodes[n].pos = (x, y, z)

        x_, y_, z_ = [], [], []
        xy_id = []
        ax = plt.axes()
        for n in nodes:
            node_ = nodes[n]
            if node_.pos == (0, 0, 0):
                x, y, z = random.uniform(0, self.graph.v_size()), random.uniform(0, self.graph.v_size()), 0
            else:
                (x, y, z) = node_.pos

            for out in self.graph.all_out_edges_of_node(n):

                o_ = nodes[out]

                if o_.pos == (0, 0, 0):
                    o_x, o_y, o_z = random.uniform(0, self.graph.v_size()), random.uniform(0, self.graph.v_size()), 0
                else:
                    (o_x, o_y, o_z) = o_.pos

                # plot arrows between nodes
                ax.quiver(x, y, o_x-x, o_y-y, angles='xy', scale_units='xy', scale=1, width=0.005)

            xy_id.append((x, y, node_.id))
            x_.append(x)
            y_.append(y)
            z_.append(z)

        # plt.xlim(left=min(x_), right=max(x_))
        # plt.ylim(bottom=min(y_), top=max(y_))

        # plot node id's
        for i, (x, y, id_) in enumerate(xy_id):
            plt.annotate(id_, (x, y), fontsize='xx-large', color='red')

        # plot nodes
        plt.plot(x_, y_, 'bo')
        plt.show()
