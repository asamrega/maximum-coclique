# distutils: language=c++

import pathlib
import networkx as nx
import time

from libcpp.vector cimport vector
from libcpp.stack cimport stack
from cython.operator cimport dereference, preincrement, postincrement


cdef extern from "<algorithm>" namespace "std":
    Iter remove[Iter, T](Iter first, Iter last, const T& value)
    OutputIt copy[InputIt, OutputIt](InputIt first, InputIt last, OutputIt d_first)
    OutputIt copy_if[InputIt, OutputIt, Pred](InputIt first, InputIt last, OutputIt d_first, Pred pred) except +

# cdef extern from "<iterator>" namespace "std":
#     back_insert_iterator[CONTAINER] back_inserter[CONTAINER](CONTAINER &)


cdef class Graph:
    cdef unsigned int _size
    cdef vector[int] _adj_matrix

    def __init__(self, size):
        self._size = size
        self._adj_matrix.resize(size*size)

    cdef int _position(self, int a, int b):
        return (a * self._size) + b

    @property
    def size(self):
        return self._size

    cpdef void add_edge(self, int a, int b):
        self._adj_matrix[self._position(a, b)] = 1
        self._adj_matrix[self._position(b, a)] = 1

    cpdef bint adjacent(self, int a, int b):
        return self._adj_matrix[self._position(a, b)]

    cpdef int degree(self, int a):
        cdef int deg = 0
        cdef int i
        for i in range(a*self._size, (a+1)*self._size):
            if self._adj_matrix[i]:
                deg += 1
        return deg


cdef void degree_sort(Graph graph, vector[int] & nodes):
    cdef int i, node
    for i, node in enumerate(sorted(range(graph.size), key=graph.degree, reverse=True)):
        nodes[i] = node


cdef void expand(Graph graph, vector[vector[int]] & buckets, vector[int] & candidate, vector[int] & neighbors, vector[int] & best):
    cdef vector[int] colors
    colors.reserve(neighbors.size())
    cdef vector[int] colored_nodes
    colored_nodes.reserve(neighbors.size())
    colourise(graph, buckets, neighbors, colors, colored_nodes)

    cdef int i, j, n
    cdef int bound, node, size
    cdef vector[int] new_neighbors
    new_neighbors.reserve(neighbors.size())
    for i in reversed(range(colored_nodes.size())):
        size = candidate.size()
        bound = size + colors[i]
        if bound <= best.size():
            return

        node = colored_nodes[i]
        candidate.push_back(node)
        postincrement(size)

        new_neighbors.clear()
        for n in neighbors:
            if graph.adjacent(node, n):
                new_neighbors.push_back(n)
        # new_neighbors = [n for n in neighbors if graph.adjacent(node, n)]
        if new_neighbors.empty():
            if size > best.size():
                best.resize(size)
                copy(candidate.begin(), candidate.end(), best.begin())
        else:
            expand(graph, buckets, candidate, new_neighbors, best)

        candidate.pop_back()
        colored_nodes.pop_back()
        colors.pop_back()
        neighbors.erase(remove(neighbors.begin(), neighbors.end(), node))


cdef colourise(Graph graph, vector[vector[int]] & buckets, vector[int] & neighbors, vector[int] & colors, vector[int] & colored_nodes):
    cdef vector[vector[int]].iterator it

    it = buckets.begin()
    while it != buckets.end():
        dereference(it).clear()
        preincrement(it)

    cdef int node, colored_node
    for node in neighbors:
        it = buckets.begin()
        while it != buckets.end():
            for colored_node in dereference(it):
                if graph.adjacent(node, colored_node):
                    break
            else:
                dereference(it).push_back(node)
                break
            preincrement(it)

    it = buckets.begin()
    while it != buckets.end():
        for node in dereference(it):
            colored_nodes.push_back(node)
            colors.push_back(it - buckets.begin() + 1)
        preincrement(it)


cpdef vector[int] search_maximum_clique(Graph graph):
    cdef vector[int] result
    result.reserve(graph.size)

    cdef vector[int] candidate
    candidate.reserve(graph.size)

    cdef vector[int] sorted_nodes
    sorted_nodes.resize(graph.size)
    degree_sort(graph, sorted_nodes)

    cdef vector[vector[int]] buckets
    cdef vector[vector[int]].iterator it
    buckets.resize(graph.size)
    it = buckets.begin()
    while it != buckets.end():
        dereference(it).reserve(graph.size)
        preincrement(it)

    expand(graph, buckets, candidate, sorted_nodes, result)
    return result


def main():
    graph_path = pathlib.Path.cwd()
    dimacs = nx.read_edgelist(graph_path.joinpath('brock200-1.txt'), nodetype=int)
    graph = Graph(dimacs.number_of_nodes())
    for i, j in dimacs.edges():
        graph.add_edge(i-1, j-1)
    x = time.time()
    result = search_maximum_clique(graph)
    print(time.time() - x)
    print(len([i+1 for i in result]))
    print([i+1 for i in result])


if __name__ == '__main__':
    main()
