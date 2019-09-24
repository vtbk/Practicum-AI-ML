import matplotlib.pyplot as plt
import random
import time
import itertools
import math
from collections import namedtuple

import heapq
# based on Peter Norvig's IPython Notebook on the TSP

City = namedtuple('City', 'x y')
Edge = namedtuple('Edge', 'p1 p2')

def distance(A, B):
    return math.hypot(A.x - B.x, A.y - B.y)

def try_all_tours(cities):
    # generate and test all possible tours of the cities and choose the shortest tour
    tours = alltours(cities)
    return min(tours, key=tour_length)

def alltours(cities):
    # return a list of tours (a list of lists), each tour a permutation of cities,
    # and each one starting with the same city
    # cities is a set, sets don't support indexing
    start = next(iter(cities)) 
    return [[start] + list(rest)
            for rest in itertools.permutations(cities - {start})]

def tour_length(tour):
    # the total of distances between each pair of consecutive cities in the tour
    return sum(distance(tour[i], tour[i-1]) 
               for i in range(len(tour)))

def make_cities(n, width=1000, height=1000):
    # make a set of n cities, each with random coordinates within a rectangle (width x height).

    random.seed(5) # the current system time is used as a seed
    # note: if we use the same seed, we get the same set of cities

    return frozenset(City(random.randrange(width), random.randrange(height))
                     for c in range(n))

def plot_tour(tour): 
    # plot the cities as circles and the tour as lines between them
    points = list(tour) + [tour[0]]
    plt.plot([p.x for p in points], [p.y for p in points], 'bo-')
    plt.axis('scaled') # equal increments of x and y have the same length
    plt.axis('off')
    plt.show()

def plot_tsp(algorithm, cities):
    # apply a TSP algorithm to cities, print the time it took, and plot the resulting tour.
    t0 = time.clock()
    tour = algorithm(cities)
    t1 = time.clock()
    print("{} city tour with length {:.1f} in {:.3f} secs for {}"
          .format(len(tour), tour_length(tour), t1 - t0, algorithm.__name__))
    print("Start plotting ...")
    plot_tour(tour)

def nearest_neighbour(cities):
    cities = set(cities)
    start = cities.pop()
    tour = [start]
    while len(cities) > 0:
        current = tour[-1]
        sorted_cities = sorted(cities, key = lambda destination : distance(current, destination))
        tour.append(sorted_cities[0])
        cities.discard(sorted_cities[0])
    return tour

def swap(tour, i, k): #https://en.wikipedia.org/wiki/2-opt
    new_tour = []
    new_tour += tour[0:i]
    middle_segment = tour[i:k + 1]
    middle_segment.reverse()
    new_tour += middle_segment
    new_tour += tour[k + 1:]
    return new_tour

def optimise(tour): #TODO Fix awful performance for larger tours (caused for the most part by the overusage of tour_length() calls)
    improved = True
    best_length = tour_length(tour)
    while improved:
        improved = False
        for i in range(0, len(tour) - 2):
            for k in range(i + 2, len(tour) - 2):
                new_tour = swap(tour, i, k)
                new_length = tour_length(new_tour)
                if new_length < best_length:
                    best_length = new_length
                    tour = new_tour
                    improved = True
    return tour

cities = make_cities(40)
tour = nearest_neighbour(cities)
optimised = optimise(tour)
print("Original cost: " + str(tour_length(tour)))
print("New cost: " + str(tour_length(optimalised)))
plot_tour(tour)
plot_tour(optimised)


#References:
#http://on-demand.gputechconf.com/gtc/2014/presentations/S4534-high-speed-2-opt-tsp-solver.pdf
#https://arthur.maheo.net/python-local-tsp-heuristics/
