import pathlib
import time

import networkx as nx


def search_maximum_clique(graph):
    maximum_clique = []
    expand(graph, [], degree_sort(graph, graph.nodes), maximum_clique)
    return maximum_clique


def expand(graph, candidate, neighbors, best):
    colors, colored_nodes = colourise(graph, neighbors)
    for i, colored_neighbor in zip(reversed(range(len(colored_nodes))), reversed(colored_nodes)):
        if len(candidate) + colors[i] > len(best):
            node = colored_neighbor
            candidate.append(node)
            new_neighbors = [n for n in neighbors if n in graph.neighbors(node)]
            if not new_neighbors:
                if len(candidate) > len(best):
                    best.clear()
                    best.extend(candidate)
            else:
                expand(graph, candidate, new_neighbors, best)
            candidate.pop()
            colored_nodes.pop()
            neighbors.remove(node)


def colourise(graph, neighbors):
    coloured_buckets = []
    for node in neighbors:
        for bucket in coloured_buckets:
            for colored_node in bucket:
                if graph.has_edge(node, colored_node):
                    break
            else:
                bucket.append(node)
                break
        else:
            coloured_buckets.append([node])

    colors = []
    colored_nodes = []
    for i, bucket in enumerate(coloured_buckets, start=1):
        for node in bucket:
            colored_nodes.append(node)
            colors.append(i)

    return colors, colored_nodes


def degree_sort(graph, nodes):
    return sorted(nodes, key=graph.degree, reverse=True)


def main():
    graph_path = pathlib.Path(pathlib.Path.cwd(), 'graphs')
    # graph = nx.algorithms.operators.unary.complement(nx.read_edgelist(graph_path.joinpath('graph2.txt')))
    graph = nx.read_edgelist(graph_path.joinpath('graph3.txt'))

    x = time.time()
    clique = search_maximum_clique(graph)
    print(time.time() - x)
    print(clique)


if __name__ == '__main__':
    main()
