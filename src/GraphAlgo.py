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
            nodes = self.graph.get_all_v()
            graph_json = {"Nodes": [], "Edges": []}

            for node_key in nodes:
                node = nodes[node_key]

                # convert pos from tuple to string
                if node.pos is None:
                    x, y, z = 0, 0, 0
                else:
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

    def bfs_util(self, v, nodes, stack):
        q = [v]
        nodes[v].tag = True

        while q:
            indx = q.pop()

            for i in self.graph.all_out_edges_of_node(indx):
                if not nodes[i].tag:
                    nodes[i].tag = True
                    q.append(i)
            stack.append(indx)

    def bfs(self, v, graph, final):
        q = [v]
        final.append(v)
        graph.nodes[v].tag = True

        while q:
            indx = heapq.heappop(q)

            for i in graph.all_out_edges_of_node(indx):

                if not graph.nodes[i].tag:
                    graph.nodes[i].tag = True
                    q.append(i)
                    final.append(i)

        return final

    def connected_component(self, id1: int) -> list:
        """
        Finds the Strongly Connected Component(SCC) that node id1 is a part of.
        used Kosaraju's algorithm with BFS

        @param id1: The node id
        @return: The list of nodes in the SCC
        """

        nodes = self.graph.get_all_v()

        for n in nodes:
            nodes[n].tag = False

        stack = []

        self.bfs_util(id1, nodes, stack)

        # transpose graph
        transpose_graph = self.graph.transpose()

        scc_path = []

        self.bfs(id1, transpose_graph, scc_path)

        return list(set(stack).intersection(scc_path))

    def connected_components(self) -> List[list]:
        """
        Finds the Strongly Connected Component(SCC) that node id1 is a part of.
        used Kosaraju's altered with BFS to avoid recursive overflow

        Finds all the Strongly Connected Component(SCC) in the graph.
        @return: The list all SCC
        """

        for n in self.graph.get_all_v():
            self.graph.get_all_v()[n].tag = False

        result = []

        for n in self.graph.get_all_v():

            flag = False
            for i in result:
                if n in i: flag = True

            if flag: continue

            for x in self.graph.get_all_v():
                self.graph.get_all_v()[x].tag = False

            stack = []

            self.bfs_util(n, self.graph.get_all_v(), stack)

            # transpose graph
            transpose_graph = self.graph.transpose()

            scc_path = []

            self.bfs(n, transpose_graph, scc_path)

            result.append(list(set(stack).intersection(scc_path)))

        return result

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """

        nodes = self.graph.get_all_v()

        for n in nodes:
            if nodes[n].pos is None:
                x, y, z = random.uniform(0, self.graph.v_size()), random.uniform(0, self.graph.v_size()), 0
                nodes[n].pos = (x, y, z)

        x_, y_, z_ = [], [], []
        xy_id = []

        ax = plt.axes()
        for n in nodes:
            node_ = nodes[n]

            (x, y, z) = node_.pos

            for out in self.graph.all_out_edges_of_node(n):

                o_ = self.graph.nodes[out]

                (o_x, o_y, o_z) = o_.pos

                # print(f"x:{x}, y:{y}, dx:{o_x-x}, dy:{o_y-y}")

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
            plt.annotate(id_, (x, y), fontsize='xx-large', color='green')

        # plot nodes
        plt.plot(x_, y_, 'ro')
        plt.show()
