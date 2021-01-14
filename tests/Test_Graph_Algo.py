import unittest
from DiGraph import DiGraph
from GraphAlgo import GraphAlgo


def build_graph(name="g"):
    print(f"Creating Graph {name}:")
    g = DiGraph()

    print("adding nodes [0,4] \n"
          "adding edges: [0,1], [0,2], [1,2], [2,3] ,[3,4]")
    for i in range(5):
        g.add_node(i)

    g.add_edge(0, 1, 1)
    g.add_edge(0, 2, 1)
    g.add_edge(1, 2, 2)

    g.add_edge(2, 3, 3)
    g.add_edge(3, 4, 5)

    return g


class Testing(unittest.TestCase):

    def test_graph(self):

        graph = build_graph()

        self.assertFalse(graph.add_node(0), "Node already exists in the graph.")

        self.assertFalse(graph.add_edge(0, 1, 1), "Edge already exists in the graph.")

        self.assertTrue(graph.remove_edge(0, 1), "Remove edge failed.")

        self.assertFalse(graph.remove_edge(0, 1), "Edge does not exist in graph.")

        self.assertFalse(graph.add_edge(0, 50, 1), "Node 50 does not exist in the graph.")

        self.assertTrue(graph.remove_node(0), "Remove node failed.")

        self.assertTrue(graph.add_node(0), "Node removal failed.")

        self.assertTrue(graph.add_edge(0, 1, 2))

        self.assertEqual(graph.__str__(), "|V|=5 , |E|=4")

        graph = build_graph()
        g_copy = build_graph("g_copy")

        self.assertEqual(g_copy.__str__(), "|V|=5 , |E|=5")

        self.assertEqual(graph, g_copy)

        self.assertTrue(g_copy.remove_node(0))

        self.assertNotEqual(graph, g_copy)

    def test_algo(self):

        graph = build_graph("graph_algo")
        algo = GraphAlgo(graph)

        print("graph: {0}".format(graph))
        print("nodes: {0}".format(graph.get_all_v()))

        # edges [0,1], [0,2], [1,2], [2,3] ,[3,4]

        self.assertEqual(algo.shortest_path(0, 4), (9, [0, 2, 3, 4]))
        self.assertEqual(algo.shortest_path(0, 2), (1, [0, 2]))
        self.assertEqual(algo.shortest_path(0, 0), (float('inf'), []))
        self.assertEqual(algo.shortest_path(2, 1), (float('inf'), []))
        self.assertEqual(algo.shortest_path(0, 7), (float('inf'), []))

        path = '../data/test_save_to_json'
        # print(f"Test: save to json -> (true, path: {path}) = {algo.save_to_json(path)}")
        self.assertTrue(algo.save_to_json(path))

        algo_load = GraphAlgo()
        # print(f"Test: load from json -> (true) = {algo_load.load_from_json(path)}")
        self.assertTrue(algo_load.load_from_json(path))

        self.assertEqual(algo.get_graph().__str__(), "|V|=5 , |E|=5")

        self.assertEqual(algo_load.connected_component(0), [0])

        algo_load.get_graph().add_edge(2, 0, 2)

        # self.assertEqual()
        self.assertEqual(algo_load.connected_component(0), [0, 2, 1])


if __name__ == '__main__':
    unittest.main()
