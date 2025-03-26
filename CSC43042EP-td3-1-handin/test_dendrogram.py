from TD.cloud import Point, Cloud, load_cloud_from_file
from TD.graph import graph_from_cloud
from TD.dendrogram import *
import sys
import time
import random

def print_point(d: Dendrogram, i: int) -> None:
    print(f'{i} ({d.get_name(i)})')

def trace_find(d: Dendrogram, i) -> int:
    s = f'{d.rank[i]}, {i} ({d.get_name(i)}), '
    p = d.parent[i]
    if p != -1:
        s += f'{p} ({d.get_name(p)}), '
    print(s)

    if p == -1:
        return i
    return trace_find(d, p)    

if __name__ == '__main__':
    print(f'== argv has {len(sys.argv)} elements')
    if len(sys.argv) < 2:
        print(f'Usage (clouds):')
        print(f'{sys.argv[0]} <filename>')
        print(f'Example: {sys.argv[0]} ./csv/iris.csv')
        print('\n')
        print(f'Usage (distance matrix):')
        print(f'{sys.argv[0]} <filename>')
        print(f'Example: {sys.argv[0]} ./csv/languages.csv')
        exit(0)
    
    with open(sys.argv[1], 'r') as infile:
        c = load_cloud_from_file(infile)
        print(f'Loaded {len(c)} points from {sys.argv[1]}')
        g = graph_from_cloud(c)

        start = time.perf_counter()
        d = Dendrogram(g)
        d.build()
        finish = time.perf_counter()
        print(f'Execution time: {finish-start}')

        print(f'Dendrogram height:\t{d.get_dendrogram_height()}')
        print('(For iris.data, height should be 0.820061)')

        print(f'Printing traces to root from 10 random points...')
        print(f'Rank, point, parent, height')
        for _ in range(10):
            node = random.randint(0, len(c) - 1)
            trace_find(d, node)
            print()

        eps = 0.01
        print(f'Looking for significant heights (up to {eps})... ',
              end='')
        d.find_heights(eps)
        print('\tdone')
        count = len(d.significant_heights)
        print(f'Found {count} significant heights (up to {eps}):')
        print(' '.join(str(h) for h in d.significant_heights))

        print('Printing clusters at significant heights')
        for (i, h_i) in enumerate(d.significant_heights):
            d.clear_clusters()
            d.set_clusters(h_i)
            print(f'{d.count_ns_clusters()} non-singleton cluster'
                  f'{"s" if d.count_ns_clusters() > 1 else ""}'
                  f' found at height {h_i}')
            d.print_clusters()
            print(d)
            print()

