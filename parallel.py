import ctypes
import pathlib
import time
from array import array
from collections import deque
from dataclasses import dataclass, field
import multiprocessing as mp
from typing import Deque

import networkx as nx


class GlobalResult:
    def __init__(self, max_size):
        self.items = mp.Array(ctypes.c_uint, max_size)
        self.size = mp.Value(ctypes.c_uint, 0)

    def merge(self, other):
        other_size = len(other)
        if other_size > self.size.value:
            self.items[:other_size] = other
            self.size.value = other_size


class WorkQueue:
    def __init__(self, number_of_dequeuers):
        self._cond = mp.Condition()
        self._queue = mp.Queue()
        self._initial_producer_done = mp.Value(ctypes.c_bool, False)
        self._donations_possible = mp.Value(ctypes.c_bool, True)
        self._want_donations = mp.Value(ctypes.c_bool, False)
        self._number_busy = mp.Value(ctypes.c_uint, number_of_dequeuers)

    def enqueue_blocking(self, item, size):
        """Called by the initial producer when producing work."""
        with self._cond:
            # Don't let the queue get too full
            while self._queue.qsize() > size:
                self._cond.wait()

            self._queue.put(item)

            # We're not empty, so we don't want donations
            self._want_donations.value = False

            # Something may be waiting in "dequeue_blocking()"
            self._cond.notify_all()

    def enqueue(self, item):
        """Called by workers when donating work."""
        with self._cond:
            self._queue.put(item)

            # We are no longer empty, so we don't want donations
            self._want_donations.value = False

            # Something may be waiting in "dequeue_blocking()"
            self._cond.notify_all()

    def dequeue_blocking(self):
        """Called by consumers waiting for work."""
        with self._cond:
            while True:
                if not self._queue.empty():  # We have something to do
                    item = self._queue.get()

                    if self._initial_producer_done and self._queue.empty():
                        # We're now empty, and the initial producer isn't
                        # going to give us anything else, so request donations.
                        self._want_donations.value = True

                    # Something might be waiting in "enqueue_blocking()"
                    self._cond.notify_all()
                    return item

                with self._number_busy.get_lock():  # We are no longer busy
                    self._number_busy.value -= 1

                if self._initial_producer_done and (not self.want_donations() or not self._number_busy.value):
                    # The queue is empty, and nothing new can possibly be
                    # produced, so we're done.  Other workers may be waiting
                    # for "_number_busy" to reach 0, so we need to wake them up.
                    self._cond.notify_all()
                    return False

                self._cond.wait()

                # We're potentially busy again
                with self._number_busy.get_lock():
                    self._number_busy.value += 1

    def initial_producer_done(self):
        """Must be called when the initial producer is finished."""
        with self._cond:
            self._initial_producer_done.value = True

            # The list might be empty, if workers dequeued quickly.
            # In that case, we now want donations.
            if self._queue.empty():
                self._want_donations.value = True

            # Maybe every worker is finished and waiting in "dequeue_blocking()"
            self._cond.notify_all()

    def want_donations(self):
        return self._donations_possible and self._want_donations.value


@dataclass(frozen=True)
class WorkQueueItem:
    candidate: Deque[int]
    neighbors: Deque[int]
    bound: int


def expand(graph, work_queue, donation_queue, candidate, neighbors, best, global_best_size):
    chose_to_donate = False  # donate work?
    colors, colored_nodes = colourise(graph, neighbors)  # Get colored nodes

    for _ in range(len(colored_nodes)):
        candidate_size = len(candidate)

        bound = candidate_size + colors.pop()
        if bound <= global_best_size.value:
            return

        node = colored_nodes.pop()
        candidate.append(node)
        candidate_size += 1
        bound += 1

        # Filter "neighbors" to contain nodes adjacent to current "node"
        new_neighbors = deque(n for n in neighbors if graph.has_edge(node, n))
        if not new_neighbors:
            if candidate_size > global_best_size.value:  # Potential new best
                global_best_size.value = candidate_size
                del best[:]
                best.extend(candidate)
        else:
            should_expand = True  # Enqueue or recurse?

            if work_queue and candidate_size == 1:  # Populate queue
                work_queue.enqueue_blocking(WorkQueueItem(candidate.copy(), new_neighbors, bound), size=6)
                should_expand = False
            elif donation_queue and (chose_to_donate or donation_queue.want_donations()):  # Donate work
                donation_queue.enqueue(WorkQueueItem(candidate.copy(), new_neighbors, bound))
                chose_to_donate = True
                should_expand = False

            if should_expand:
                expand(graph, work_queue, donation_queue, candidate, new_neighbors, best, global_best_size)

        candidate.pop()
        neighbors.remove(node)


def colourise(graph, neighbors):
    """Perform color sorting."""
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
    """Sort nodes in non-increasing degree order."""
    return deque(sorted(graph.nodes, key=graph.degree, reverse=True))


def populator(graph, work_queue, sorted_nodes, global_best, global_best_size):
    local_best = array('I')  # Local result

    # Do initial population
    expand(graph,
           work_queue=work_queue,
           donation_queue=None,
           candidate=deque(),
           neighbors=sorted_nodes,
           best=local_best,
           global_best_size=global_best_size)

    # Signal that the initial producer is finished
    work_queue.initial_producer_done()

    global_best.merge(local_best)


def worker(graph, donation_queue, global_best, global_best_size):
    local_best = array('I')  # Local result

    while True:
        item = donation_queue.dequeue_blocking()  # Get some work to do
        if not item:
            break

        # Re-evaluate the bound against our new best
        if item.bound <= global_best_size.value:
            continue

        # Do some work
        expand(graph,
               work_queue=None,
               donation_queue=donation_queue,
               candidate=item.candidate,
               neighbors=item.neighbors,
               best=local_best,
               global_best_size=global_best_size)

    global_best.merge(local_best)


def search_maximum_clique(graph):
    global_best = GlobalResult(graph.number_of_nodes())  # Global result
    global_best_size = mp.Value(ctypes.c_uint, 0)  # Shared size of the global result

    number_of_workers = mp.cpu_count()
    work_queue = WorkQueue(number_of_workers)

    sorted_nodes = degree_sort(graph)

    # Create populating process
    p = mp.Process(target=populator, args=(graph, work_queue, sorted_nodes, global_best, global_best_size))
    p.start()

    # Create working processes
    workers = []
    for _ in range(number_of_workers):
        w = mp.Process(target=worker, args=(graph, work_queue, global_best, global_best_size))
        workers.append(w)
        w.start()

    # Wait until they are done
    p.join()
    for w in workers:
        w.join()

    return global_best.items[:global_best_size.value]


def main():
    print('Number of processors: ', mp.cpu_count())

    graph_path = pathlib.Path(pathlib.Path.cwd(), 'graphs')
    # graph = nx.algorithms.operators.unary.complement(nx.read_edgelist(graph_path.joinpath('graph2.txt')))
    graph = nx.read_edgelist(graph_path.joinpath('graph3.txt'), nodetype=int)

    x = time.time()
    clique = search_maximum_clique(graph)
    print(time.time() - x)

    print(clique)


if __name__ == '__main__':
    main()
