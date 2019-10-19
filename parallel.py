import ctypes
import pathlib
from array import array
import multiprocessing as mp
from collections import deque

import networkx as nx


class WorkQueue:
    def __init__(self, number_of_dequeuers):
        self._cond = mp.Condition()
        self._donations_possible = mp.Value(ctypes.c_bool, True)
        self._queue = mp.Queue()
        self._initial_producer_done = mp.Value(ctypes.c_bool, False)
        self._want_donations = mp.Value(ctypes.c_bool, False)
        self._number_busy = mp.Value(ctypes.c_uint, number_of_dequeuers)

    def enqueue_blocking(self, *item, size):
        with self._cond:
            while self._queue.qsize() > size:
                self._cond.wait()

            self._queue.put(item)
            self._cond.notify_all()

    def dequeue_blocking(self, item):
        with self._cond:
            while True:
                if not self._queue.empty():
                    item.extend(self._queue.get())

                    if self._initial_producer_done and self._queue.empty():
                        self._want_donations.value = True

                    self._cond.notify_all()
                    return True

                with self._number_busy.get_lock():
                    self._number_busy.value -= 1

                if self._initial_producer_done and (not self.want_donations() or not self._number_busy.value):
                    self._cond.notify_all()
                    return False

                self._cond.wait()

                with self._number_busy.get_lock():
                    self._number_busy.value += 1

    def initial_producer_done(self):
        with self._cond:
            with self._initial_producer_done.get_lock():
                self._initial_producer_done.value = True

            if self._queue.empty():
                self._want_donations.value = True

            self._cond.notify_all()

    def want_donations(self):
        return self._donations_possible and self._want_donations.value


class MaxCliqueResult:
    def __init__(self, max_size):
        self.members = mp.Array(ctypes.c_uint, max_size)
        self.size = mp.Value(ctypes.c_uint, 0)

    def merge(self, other):
        other_size = len(other)
        if other_size > self.size.value:
            self.size.value = other_size
            for i in range(other_size):
                self.members[i] = other[i]


def populator(graph, global_result, global_best, work_queue, sorted_nodes):
    result = array('I')
    candidate = deque()
    neighbors = sorted_nodes.copy()
    expand(graph, work_queue, None, candidate, neighbors, result, global_best)
    # work_queue.initial_producer_done()
    global_result.merge(result)
    print([i for i in global_result.members])


def worker(graph, donation_queue, global_result, global_best):
    result = array('I')
    name = mp.current_process().name
    # print(f'Start worker {name!r}')
    while True:
        item = []
        if not donation_queue.dequeue_blocking(item):
            break

        print(item)
        if item[2] <= global_best.value:
            continue
        # expand(graph, None, donation_queue, item[0], item[1], result, global_best)
        # global_result.merge(result)


def search_maximum_clique(graph):
    global_result = MaxCliqueResult(graph.number_of_nodes())
    global_best = mp.Value(ctypes.c_uint, 0)
    work_queue = WorkQueue(mp.cpu_count())  # TODO: make work queue
    sorted_nodes = degree_sort(graph)

    p = mp.Process(target=populator, args=(graph, global_result, global_best, work_queue, sorted_nodes))
    p.start()

    workers = [mp.Process(target=worker, args=(graph, work_queue, global_result, global_best)) for _ in range(6)]
    for w in workers:
        w.start()

    p.join()
    for w in workers:
        w.join()


def expand(graph, work_queue, donation_queue, candidate, neighbors, result, global_best):
    chose_to_donate = False
    colors, colored_nodes = colourise(graph, neighbors)
    print(colored_nodes)
    for _ in range(len(colored_nodes)):
        bound = colors.pop()
        candidate_size = len(candidate)
        if candidate_size + bound <= global_best.value:
            return

        node = colored_nodes.pop()
        candidate.append(node)
        candidate_size += 1

        new_neighbors = deque(n for n in neighbors if graph.has_edge(node, n))
        if not new_neighbors:
            if candidate_size > global_best.value:
                global_best.value = candidate_size
                del result[:]
                result.extend(candidate)
        else:
            should_expand = True

            if work_queue and candidate_size == 1:
                work_queue.enqueue_blocking(candidate, new_neighbors, candidate_size+bound, size=6)  # TODO: check arguments
                # print(f'Enqueue: {candidate}, {new_neighbors}, {candidate_size+bound}')  # TODO: check arguments
                should_expand = False
            elif donation_queue and (chose_to_donate or donation_queue.want_donations()):
                # donation_queue.enqueue(candidate, new_neighbors, candidate_size+bound)   # TODO: check arguments
                should_expand = False
                chose_to_donate = True

            if should_expand:
                expand(graph, work_queue, donation_queue, candidate, new_neighbors, result, global_best)

        candidate.pop()
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

    colors = deque()
    colored_nodes = deque()
    for i, bucket in enumerate(coloured_buckets, start=1):
        for node in bucket:
            colored_nodes.append(node)
            colors.append(i)

    return colors, colored_nodes


def degree_sort(graph):
    return deque(sorted(graph.nodes, key=graph.degree, reverse=True))


def main():
    print('Number of processors: ', mp.cpu_count())

    graph_path = pathlib.Path(pathlib.Path.cwd(), 'graphs')
    # graph = nx.algorithms.operators.unary.complement(nx.read_edgelist(graph_path.joinpath('graph2.txt')))
    graph = nx.read_edgelist(graph_path.joinpath('graph1.txt'), nodetype=int)

    search_maximum_clique(graph)


if __name__ == '__main__':
    main()
