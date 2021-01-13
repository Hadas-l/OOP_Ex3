import unittest
import time
import timeit
from GraphAlgo import GraphAlgo
import random
import sys
import networkx
import json


class Tests(unittest.TestCase):

    def test_python(self):

        result = {"run_time": [], "sp_time": [], "scc_time": []}

        G_10_80_1 = '../data/G_10_80_1.json'
        G_100_800_1 = '../data/G_100_800_1.json'
        G_1000_8000_1 = '../data/G_1000_8000_1.json'
        G_10000_80000_1 = '../data/G_10000_80000_1.json'
        G_20000_160000_1 = '../data/G_20000_160000_1.json'
        G_30000_240000_1 = '../data/G_30000_240000_1.json'

        file_list = [G_10_80_1,
                     G_100_800_1,
                     G_1000_8000_1,
                     G_10000_80000_1,
                     G_20000_160000_1,
                     G_30000_240000_1]

        algo = GraphAlgo()

        print("load execution times:")

        for file_name in file_list:

            start_timeit = timeit.default_timer()
            start_time = time.perf_counter()

            # execute logic
            self.assertTrue(algo.load_from_json(file_name), "load failed.")

            stop_timeit = timeit.default_timer() - start_timeit
            stop_time = time.perf_counter() - start_time

            # print("file: {}, by timeit: {}, by time: {}".format(file_name, stop_timeit, stop_time))
            result['run_time'].append({file_name: (stop_time + stop_timeit) / 2})

        print(result['run_time'])

        print("shortest path execution time")

        for file_name in file_list:

            start_timeit = timeit.default_timer()
            start_time = time.perf_counter()

            # execute logic
            self.assertTrue(algo.load_from_json(file_name), "load failed.")
            s = random.randint(0, algo.get_graph().v_size())
            dest = random.randint(0, algo.get_graph().v_size())
            d, path = algo.shortest_path(s, dest)
            # print("Algo from {} to {}: {}, {}".format(0, 5, d, path))

            stop_timeit = timeit.default_timer() - start_timeit
            stop_time = time.perf_counter() - start_time

            # print("file + Algo: {}, by timeit: {}, by time: {}".format(file_name, stop_timeit, stop_time))
            result['sp_time'].append({file_name: (stop_time+stop_timeit)/2})

        print(result['sp_time'])

        print("SCC execution time")

        for file_name in file_list:

            start_timeit = timeit.default_timer()
            start_time = time.perf_counter()

            # execute logic
            self.assertTrue(algo.load_from_json(file_name), "load failed.")

            # algo.connected_components()
            # algo.connected_component(random.randint(0, algo.get_graph().v_size()))

            stop_timeit = timeit.default_timer() - start_timeit
            stop_time = time.perf_counter() - start_time

            result['scc_time'].append({file_name: (stop_time + stop_timeit) / 2})

        print(result['scc_time'])

    def test_netowrkx(self):

        result = {"nx_run_time": [], "nx_sp_time": [], "nx_scc_time": []}

        G_10_80_1 = '../data/G_10_80_1.json'
        G_100_800_1 = '../data/G_100_800_1.json'
        G_1000_8000_1 = '../data/G_1000_8000_1.json'
        G_10000_80000_1 = '../data/G_10000_80000_1.json'
        G_20000_160000_1 = '../data/G_20000_160000_1.json'
        G_30000_240000_1 = '../data/G_30000_240000_1.json'

        file_list = [G_10_80_1,
                     G_100_800_1,
                     G_1000_8000_1,
                     G_10000_80000_1,
                     G_20000_160000_1,
                     G_30000_240000_1]

        algo_dict = {G_10_80_1: None,
                     G_100_800_1: None,
                     G_1000_8000_1: None,
                     G_20000_160000_1: None,
                     G_30000_240000_1: None}

        print("load netowrkx execution times:")

        for file_name in file_list:

            start_timeit = timeit.default_timer()
            start_time = time.perf_counter()

            # init netowrkx graph
            nx_graph = networkx.DiGraph()

            # load from json to networkx
            edges = []
            nodes = []

            with open(file_name) as file:
                data = json.load(file)

            for e in data['Edges']:
                edges.append((e['src'], e['dest'], e['w']))

            for n in data['Nodes']:
                nodes.append(n['id'])

            # add nodes by list of id's [0, 1, 2 , ... ]
            # add edges by list of tuples [(s1, d1, w1), (s2, d2, w2), .... ]
            nx_graph.add_nodes_from(nodes)
            nx_graph.add_weighted_edges_from(edges)

            stop_timeit = timeit.default_timer() - start_timeit
            stop_time = time.perf_counter() - start_time

            # print("networkx load file: {}, by timeit: {}, by time: {}".format(file_name, stop_timeit, stop_time))
            result['nx_run_time'].append({file_name: (stop_time + stop_timeit) / 2})

            algo_dict[file_name] = nx_graph

        print(result['nx_run_time'])

        print("networkx shortest path execution time:")

        for file_name in file_list:

            start_timeit = timeit.default_timer()
            start_time = time.perf_counter()

            # execute networkx logic
            path = []

            s = random.randint(0, algo_dict[file_name].number_of_nodes()),
            d = random.randint(0, algo_dict[file_name].number_of_nodes())

            try:
                path = networkx.shortest_path(algo_dict[file_name], 0, 5)
            except:
                pass

            stop_timeit = timeit.default_timer() - start_timeit
            stop_time = time.perf_counter() - start_time

            result['nx_sp_time'].append({file_name: (stop_time + stop_timeit) / 2})

        print(result['nx_sp_time'])

        print("networkx scc+ time:")

        for file_name in file_list:

            start_timeit = timeit.default_timer()
            start_time = time.perf_counter()

            # execute networkx logic

            try:
                a = networkx.strongly_connected_components(algo_dict[file_name])
            except:
                pass

            stop_timeit = timeit.default_timer() - start_timeit
            stop_time = time.perf_counter() - start_time

            # print("networkx scc: {}, by timeit: {}, by time: {}".format(file_name, stop_timeit, stop_time))
            result['nx_scc_time'].append({file_name: (stop_time + stop_timeit) / 2})

        print(result['nx_scc_time'])


if __name__ == '__main__':
    sys.setrecursionlimit(10**6)
    unittest.main()
