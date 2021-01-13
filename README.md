# OOP_Ex3

assignment 3, Ariel University, OOP Course 2020-2021.

## Introduction

in this assignment I will present a Python implementation of our Java 2nd assignment.
in Ex2 we implemented a bi-directional graph using HashMaps in java.

## requirements

must install matplotlib, and networkx for testing.

```bash
pip install matplotlib
pip install networkx
```

## usage

one can use the library for a bi-directional graph. <br/>
also in this library you can find algorithms for your graph such as: <br/>
*shortest path <br/>
*save to json <br/>
*load from json <br/>

in python:

```python

graph = DiGraph()

graph.add_node(0)
graph.add_node(1)

# add edge with src -> dest -> weight
graph.add_edge(0, 1, 2)

algo = GraphAlgo(graph)

algo.save_to_json(file_path)
algo.load_from_json(file_path)

shortest_path, path = algo.shortest_path(0, 5)
```

## project goal

the goal for the project is to compare execution times for project code, networkx, jave ex2 code
for several functions such as: <br/>
* load from json
* shortest path
* strongly connected components


[picture](images/graph.png)

raw data can be found in the wiki page.
