#!/usr/bin/env python
# coding: utf-8

# # Libraries

# * import the modules that will be needed in this program

# In[378]:


import sys
from collections import deque
from collections import OrderedDict
from copy import deepcopy
from tkinter import *
from PIL import Image, ImageTk
from utils import *
import heapq
import memoize 
import numpy as np


# ## Problem

# In the following cell, we are defining a problem class that will be used to solve the traveling salesman problem (finding the optimal distance). The class has the following methods:
# * **actions**, which returns the actions that can be executed in a state
# * **result**, which returns the state that results from executing an action in a given state
# * **goal_test**, which compares the current state to the goal and returns true if this state is the goal
# * **path_cost**, which returns the cost of the path
# * **value**, which returns the value that we are trying to optimize

# In[379]:


class Problem:

    def __init__(self, initial, goal=None):
        """The constructor """
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value"""
        raise NotImplementedError


# ## Node Class
# This class creates a search tree node that is derived from a parent by an action. The class has the following methods:
# * **repr**, which returns the current state of our problem
# * **it**, which shows whether the current state is less than the node state
# * **expand**, which lists the nodes reachable in one step from this node
# * **child_node**, which returns the children of our current node
# * **solution**, which returns the sequence of actions to go from the root to this node
# * **path**, which returns a list of nodes forming the path from the root to this node
# * **eq**, which treats nodes with the same state as equal
# * **hash**, which returns nodes with the same state

# In[380]:


class Node:

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return f"<Node {self.state}>"

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # a queue of nodes in breadth_first_graph_search or
    # astar_tree_search to have no duplicated states, so treat nodes
    # with the same state as equal.

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


# ## Depth First Search
# This function performs a depth first search on our problem. The function has two variables, frontier and explore, which are of type stack and set respectively. Frontier stores the initial problem nodes while explored stores a set of nodes that have already been searched.
# 

# In[381]:


def depth_first_graph_search(problem):

    frontier = [(Node(problem.initial))]  # Stack

    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return None


# ## Breadth First Search
# 

# In[382]:


def breadth_first_graph_search(problem):

    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None


# # PriorityQueue Class

# In[383]:


class priorityQueue:
    def __init__(self):
        self.cities = []

    def push(self, city, cost):
        heapq.heappush(self.cities, (cost, city))

    def pop(self):
        return heapq.heappop(self.cities)[1]

    def isEmpty(self):
        if (self.cities == []):
            return True
        else:
            return False

    def check(self):
        print(self.cities)


# ## best_first_graph_search Function

# In[384]:


# def best_first_graph_search(problem, f, display=False):

#     f = memoize(f, 'f')
#     node = Node(problem.initial)
#     frontier = priorityQueue()
#     frontier.append(node)
#     explored = set()
#     while frontier:
#         node = frontier.pop()
#         if problem.goal_test(node.state):
#             if display:
#                 print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
#             return node
#         explored.add(node.state)
#         for child in node.expand(problem):
#             if child.state not in explored and child not in frontier:
#                 frontier.append(child)
#             elif child in frontier:
#                 if f(child) < frontier[child]:
#                     del frontier[child]
#                     frontier.append(child)
#     return None


# 

# ## A* Algorithm Function
# 
# This function uses straight line distance (Euclidean Distance) as a heuristic. 
# * We get the straight line distances of all nodes in the graph to the goal node. This is the h() value.
# * The g() value is the distance from the current node to one of it's neighbours
# * f() = g() + h()
# * the lower the f value, the higher up on the priority queue. It will be removed first

# In[385]:


def astar_tree_search(problem):
    global frontier, counter, node
    if counter == -1:
        frontier = deque()

    if counter == -1:
        frontier.append(Node(problem.initial))

        display_frontier(frontier)
    if counter % 3 == 0 and counter >= 0:
        node = frontier.popleft()

        display_current(node)
    if counter % 3 == 1 and counter >= 0:
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))

        display_frontier(frontier)
    if counter % 3 == 2 and counter >= 0:
        display_explored(node)
        return None;
   
    
#     graph = Graph.Graph()
#     create_a_star_graph(graph)

#     if start not in graph.nodeNames or end not in graph.nodeNames:
#         print("Unknown Town")
#         return 0

#     start = graph.setStart(start)
#     end = graph.setEnd(end)

    
#     open_list = P_Queue.PriorityQueue() 
#     start.setG(0)
#     start.setH(getSLD(start.value, end.value))
#     start.searched = True
#     open_list.addElem((start.setF(start.g + start.h), start)) #at the beginning, the g value is 0.
#     closed_list = [] #holds the names of the towns

#     attempts = [] #array to hold how all paths were tried. 

#     #open_list elements are a 2-tuple of the form (f(), NodeObject)

#     while len(open_list.queue) != 0: 
#         closed_list.append(open_list.queue[0][1].value)
#         # print(open_list.queue[0][1].value , {"f: ": open_list.queue[0][1].f , "g: ": open_list.queue[0][1].g, "h: ": open_list.queue[0][1].h} )
#         current = open_list.pop()[1]
        
#         #update the attempts sequence
#         if current.parent == None:
#             attempts.append(townDisplay(current.value, None))
#         else: 
#             attempts.append(townDisplay(current.value, current.parent.value))

#         if current.value == end.value:
#             # print("Found Target:", current.value)
#             break

#         edges = current.edges
#         for edge in edges:
#             neighbour = graph.getNode(list(edge.keys())[0]) #get the neighbour node at the end of this edge
            
#             if not neighbour.searched:
#                 neighbour.searched = True
#                 neighbour.parent = current
#                 h = getSLD(neighbour.value, end.value )
#                 g = neighbour.setCummulativeG(current.getEdgeDistance(neighbour.value))
#                 f = h+g
#                 neighbour.setF(f)
#                 neighbour.setG(g)
#                 neighbour.setH(h)
                
#                 neighbour.current = current.value
#                 neighbour.parent = current
#                 open_list.addElem((neighbour.f ,neighbour))
#                 #print(open_list.queue[0][1].value, open_list.queue[0][0] )

#     path = []
#     path.append(end)
#     nextNode = end.parent
#     while nextNode != None:
#         path.append(nextNode)
#         nextNode = nextNode.parent
    
#     txt = ""
#     for i in range(len(path))[::-1]:
#         n = path[i]
#         txt += n.value
#         if i != 0:
#             txt += " --> "
            
#     print("Path using AStar :: " ,txt)
#     #print("A Start Attempts: ", attempts)
#     return attempts, txt


# # Graph Class
# 
# A state space is a graph (V, E) where V is a set of nodes and E is a set of arcs, and each arc is directed from a node to another node.

# In[386]:


class Graph:

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


# ## UndirectedGraph Function
# 
# We build a Graph where every edge (including future ones) goes both ways.

# In[387]:


def UndirectedGraph(graph_dict=None):
   
    return Graph(graph_dict=graph_dict, directed=False)


# ## RandomGraph Function

# This function builds the cities and the minimum links to the nearest neighbors

# In[388]:


# def RandomGraph(nodes=list(range(10)), min_links=2, width=400, height=300,
#                 curvature=lambda: random.uniform(1.1, 1.5)):
#     g = UndirectedGraph()
#     g.locations = {}
#     # Build the cities
#     for node in nodes:
#         g.locations[node] = (random.randrange(width), random.randrange(height))
#     # Build roads from each city to at least min_links nearest neighbors.
#     for i in range(min_links):
#         for node in nodes:
#             if len(g.get(node)) < min_links:
#                 here = g.locations[node]

#                 def distance_to_node(n):
#                     if n is node or g.get(node, n):
#                         return np.inf
#                     return distance(g.locations[n], here)

#                 neighbor = min(nodes, key=distance_to_node)
#                 d = distance(g.locations[neighbor], here) * curvature()
#                 g.connect(node, neighbor, int(d))
#     return g


# In[389]:


k_map = UndirectedGraph(dict(
    Nairobi=dict(Mombasa = 490, Meru=225, Kisii=310, Nakuru=160,Machakos=69,Lamu=467),
    Kisumu=dict(Kisii=71),
    Lamu=dict(Mombasa=240),
    Machakos=dict(Mombasa=399),
    Nakuru =dict(Kisii = 194, Eldoret=156),
    Eldoret=dict(Kitale = 72),
    Kitale =dict(Kisii=264)))

k_map.locations = dict(
   Kitale=(108,606), Eldoret=(131,561),
   Kisumu=(50,370),
   Nakuru=(207,457), Nairobi=(233,390),
   Kisii=(74,430),Meru=(305,449),
   Machakos=(300,300),
   Lamu=(550,300),
   Mombasa=(562,193))

root = None
city_coord = {}
k_problem = None
algo = None
start = None
goal = None
counter = -1
city_map = None
frontier = None
front = None
node = None
next_button = None
explored = None


# ## create_map function
# 
# This function draws out the required map

# In[390]:


def create_map(root):
    #draw map
    global city_map, start, goal
    k_locations =k_map.locations
    width = 750
    height = 670
    margin = 5
    city_map = Canvas(root, width=width, height=height)
    city_map.pack()

    # Since lines have to be drawn between particular points, we need to list
    # them separately
    make_line(
        city_map,
        k_locations['Kitale'][0],
        height -
        k_locations['Kitale'][1],
        k_locations['Eldoret'][0],
        height -
        k_locations['Eldoret'][1],
        k_map.get('Kitale', 'Eldoret'))
    make_line(
        city_map,
        k_locations['Nairobi'][0],
        height -
        k_locations['Nairobi'][1],
        k_locations['Mombasa'][0],
        height -
        k_locations['Mombasa'][1],
        k_map.get('Nairobi', 'Mombasa'))
    make_line(
        city_map,
        k_locations['Nairobi'][0],
        height -
        k_locations['Nairobi'][1],
        k_locations['Meru'][0],
        height -
        k_locations['Meru'][1],
        k_map.get('Nairobi','Meru'))
    make_line(
        city_map,
        k_locations['Nairobi'][0],
        height -
        k_locations['Nairobi'][1],
        k_locations['Kisii'][0],
        height -
        k_locations['Kisii'][1],
        k_map.get('Nairobi', 'Kisii'))
    make_line(
        city_map,
        k_locations['Nairobi'][0],
        height -
        k_locations['Nairobi'][1],
        k_locations['Nakuru'][0],
        height -
        k_locations['Nakuru'][1],
        k_map.get('Nairobi', 'Nakuru'))
    make_line(
        city_map,
        k_locations['Nakuru'][0],
        height -
        k_locations['Nakuru'][1],
        k_locations['Kisii'][0],
        height -
        k_locations['Kisii'][1],
        k_map.get('Nakuru', 'Kisii'))
    make_line(
        city_map,
        k_locations['Nakuru'][0],
        height -
        k_locations['Nakuru'][1],
        k_locations['Eldoret'][0],
        height -
        k_locations['Eldoret'][1],
        k_map.get('Nakuru', 'Eldoret'))
    make_line(
        city_map,
        k_locations['Kitale'][0],
        height -
        k_locations['Kitale'][1],
        k_locations['Kisii'][0],
        height -
        k_locations['Kisii'][1],
        k_map.get('Kitale', 'Kisii'))
    make_line(
        city_map,
        k_locations['Kisii'][0],
        height -
        k_locations['Kisii'][1],
        k_locations['Kisumu'][0],
        height -
        k_locations['Kisumu'][1],
        k_map.get('Kisii', 'Kisumu'))
    
    make_line(
        city_map,
        k_locations['Nairobi'][0],
        height -
        k_locations['Nairobi'][1],
        k_locations['Machakos'][0],
        height -
        k_locations['Machakos'][1],
        k_map.get('Nairobi', 'Machakos'))
    make_line(
        city_map,
        k_locations['Machakos'][0],
        height -
        k_locations['Machakos'][1],
        k_locations['Mombasa'][0],
        height -
        k_locations['Mombasa'][1],
        k_map.get('Machakos', 'Mombasa'))
    make_line(
        city_map,
        k_locations['Nairobi'][0],
        height -
        k_locations['Nairobi'][1],
        k_locations['Lamu'][0],
        height -
        k_locations['Lamu'][1],
        k_map.get('Nairobi', 'Lamu'))
    make_line(
        city_map,
        k_locations['Mombasa'][0],
        height -
        k_locations['Mombasa'][1],
        k_locations['Lamu'][0],
        height -
        k_locations['Lamu'][1],
        k_map.get('Mombasa', 'Lamu'))
   

    for city in k_locations.keys():
        make_rectangle(
            city_map,
            k_locations[city][0],
            height -
            k_locations[city][1],
            margin,
            city)

    make_legend(city_map)


# ## make_line function
# 
# This function draws out the lines joining various points.

# In[391]:


def make_line(map, x0, y0, x1, y1, distance):
 
    map.create_line(x0, y0, x1, y1)
    map.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=distance)


# ## make_rectangle
# 
# This function draws rectangles at each node (city)

# In[392]:


def make_rectangle(map, x0, y0, margin, city_name):
    """This function draws out rectangles for various points."""
    global city_coord
    rect = map.create_rectangle(
        x0 - margin,
        y0 - margin,
        x0 + margin,
        y0 + margin,
        fill="white")
    if "Nakuru" in city_name or "Nairobi" in city_name or "Kisii" in city_name             or "KItale" in city_name or "Mombasa" in city_name:
        map.create_text(
            x0 - 2 * margin,
            y0 - 2 * margin,
            text=city_name,
            anchor=E)
    else:
        map.create_text(
            x0 - 2 * margin,
            y0 - 2 * margin,
            text=city_name,
            anchor=SE)
    city_coord.update({city_name: rect})


# ## make_legend
# 
# create a legend to enable users to identify the un-explored, frontier, currently exploring, explored, and final solution nodes.

# In[393]:


def make_legend(map):
    rect1 = map.create_rectangle(600, 100, 610, 110, fill="white")
    text1 = map.create_text(615, 105, anchor=W, text="Un-explored")

    rect2 = map.create_rectangle(600, 115, 610, 125, fill="orange")
    text2 = map.create_text(615, 120, anchor=W, text="Frontier")

    rect3 = map.create_rectangle(600, 130, 610, 140, fill="green")
    text3 = map.create_text(615, 135, anchor=W, text="Currently Exploring")

    rect4 = map.create_rectangle(600, 145, 610, 155, fill="grey")
    text4 = map.create_text(615, 150, anchor=W, text="Explored Already")

    rect5 = map.create_rectangle(600, 160, 610, 170, fill="red")
    text5 = map.create_text(615, 165, anchor=W, text="Final Solution")
    


# ## graph_search
# This function searches through the successors of a problem to find a goal.
#     The argument frontier should be an empty queue.

# In[394]:


# def graph_search(problem):
#     """
    
#    """
#     global counter, frontier, node, explored
#     if counter == -1:
#         frontier.append(Node(problem.initial))
#         explored = set()

#         display_frontier(frontier)
#     if counter % 3 == 0 and counter >= 0:
#         node = frontier.pop()

#         display_current(node)
#     if counter % 3 == 1 and counter >= 0:
#         if problem.goal_test(node.state):
#             return node
#         explored.add(node.state)
#         frontier.extend(child for child in node.expand(problem)
#                         if child.state not in explored and
#                         child not in frontier)

#         display_frontier(frontier)
#     if counter % 3 == 2 and counter >= 0:
#         display_explored(node)
#     return None


# ## display_frontier
# This function marks the frontier nodes (orange) on the map.

# In[395]:


def display_frontier(queue):
    
    global city_map, city_coord
    qu = deepcopy(queue)
    while qu:
        node = qu.pop()
        for city in city_coord.keys():
            if node.state == city:
                city_map.itemconfig(city_coord[city], fill="orange")


# ## display_current
# This function marks the currently exploring node (red) on the map

# In[396]:


def display_current(node):
   
    global city_map, city_coord
    city = node.state
    city_map.itemconfig(city_coord[city], fill="green")


# ## display_explored
# This function marks the already explored node (gray) on the map.

# In[397]:


def display_explored(node):
   
    global city_map, city_coord
    city = node.state
    city_map.itemconfig(city_coord[city], fill="gray")


# ## display_final
# This function marks the final solution nodes (green) on the map.

# In[398]:


def display_final(cities):
    
    global city_map, city_coord
    for city in cities:
        city_map.itemconfig(city_coord[city], fill="red")


# ## breadth_first_graph search
# Search the tree by expanding each level

# In[399]:


def breadth_first_graph_search(problem):
    
    global frontier, node, explored, counter
    if counter == -1:
        node = Node(problem.initial)
        display_current(node)
        if problem.goal_test(node.state):
            return node

        frontier = deque([node])  # FIFO queue

        display_frontier(frontier)
        explored = set()
    if counter % 3 == 0 and counter >= 0:
        node = frontier.popleft()
        display_current(node)
        explored.add(node.state)
    if counter % 3 == 1 and counter >= 0:
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
        display_frontier(frontier)
    if counter % 3 == 2 and counter >= 0:
        display_explored(node)
    return None


# ## depth_first_graph_search
# Search the deepest nodes in the search tree first.

# In[400]:


def depth_first_graph_search(problem):
    
    global counter, frontier, node, explored
    if counter == -1:
        frontier = []  # stack
    if counter == -1:
        frontier.append(Node(problem.initial))
        explored = set()

        display_frontier(frontier)
    if counter % 3 == 0 and counter >= 0:
        node = frontier.pop()

        display_current(node)
    if counter % 3 == 1 and counter >= 0:
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and
                        child not in frontier)

        display_frontier(frontier)
    if counter % 3 == 2 and counter >= 0:
        display_explored(node)
    return None


# # GraphProblem Class
# 

# In[401]:


class GraphProblem(Problem):
    #The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        #The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        #The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        #Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        #h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf


# # GraphProblemStochastic Class
# 
# This function gets the states and actions of a given search problem
# 

# In[402]:


class GraphProblemStochastic(GraphProblem):

    def result(self, state, action):
        return self.graph.get(state, action)

    def path_cost(self):
        raise NotImplementedError


# ## on_click function
# 
# This function defines the action of the 'Next' button.

# In[403]:


def on_click():
    
    global algo, counter, next_button, k_problem, start, goal
    k_problem = GraphProblem(start.get(), goal.get(), k_map)
    if "Breadth-First Graph Search" == algo.get():
        node = breadth_first_graph_search(k_problem)
        if node is not None:
            final_path = breadth_first_graph_search(k_problem).solution()
            final_path.append(start.get())
            display_final(final_path)
            test_path = final_path[-1:] + final_path[:-1] 
            summation=0
            for i in range(len(test_path)-1):
                x = test_path[i]
                y = test_path[i+1]
                
                summation=summation + int(k_map.get(x, y))
            print(summation)
            next_button.config(state="disabled")
        counter += 1
    elif "Depth-First Graph Search" == algo.get():
        node = depth_first_graph_search(k_problem)
        if node is not None:
            final_path = depth_first_graph_search(k_problem).solution()
            final_path.append(start.get())
            display_final(final_path)
            test_path = final_path[-1:] + final_path[:-1] 
            summation=0
            for i in range(len(test_path)-1):
                x = test_path[i]
                y = test_path[i+1]
                
                summation=summation + int(k_map.get(x, y))
            print(summation)
            next_button.config(state="disabled")
        counter += 1
    elif "A* - Search" == algo.get():
        node = astar_tree_search(k_problem)
        if node is not None:
            final_path = astar_tree_search(k_problem).solution()
            final_path.append(start.get())
            display_final(final_path)
            print(final_path)
            test_path = final_path[-1:] + final_path[:-1] 
            summation=0
            for i in range(len(test_path)-1):
                x = test_path[i]
                y = test_path[i+1]
                
                summation=summation + int(k_map.get(x, y))
            print(summation)
            next_button.config(state="disabled")
        counter += 1


# ## reset_map function

# This function resets the map after performing a search

# In[404]:


def reset_map():
    global counter, city_coord, city_map, next_button
    counter = -1
    for city in city_coord.keys():
        city_map.itemconfig(city_coord[city], fill="white")
    next_button.config(state="normal")


# ## Execute the problem : Perform the Search

# In[ ]:


if __name__ == "__main__":
    
    algo, start, goal, next_button
    root = Tk()
    #root = tk.Tk()
    #gui = Gui()
    # test()
    #root.mainloop()
    root.title("DFS,BFS,and A* Route Finding")
    root.geometry("950x1150")
    """
    bg= ImageTk.PhotoImage(file="./ye.jpg")
    #Create a canvas
    canvas= Canvas(width= 400, height= 200)
    canvas.pack(expand=True, fill= BOTH)
    canvas.create_image(0,0,image=bg, anchor="nw")

    """
    algo = StringVar(root)
    start = StringVar(root)
    goal = StringVar(root)
    algo.set("Breadth-First Graph Search")
    start.set('Kitale')
    goal.set('Mombasa')
    cities = sorted(k_map.locations.keys())
    algorithm_menu = OptionMenu(
        root,
        algo, 
        "Breadth-First Graph Search", "Depth-First Graph Search", "A* - Search")
    Label(root, text="\n Search Algorithm").pack()
    algorithm_menu.pack()
    Label(root, text="\n Start City").pack()
    start_menu = OptionMenu(root, start, *cities)
    start_menu.pack()
    Label(root, text="\n Goal City").pack()
    goal_menu = OptionMenu(root, goal, *cities)
    goal_menu.pack()
    frame1 = Frame(root)
    next_button = Button(
        frame1,
        width=6,
        height=2,
        text="Next",
        command=on_click,
        padx=2,
        pady=2,
        relief=GROOVE)
    next_button.pack(side=RIGHT)
    reset_button = Button(
        frame1,
        width=6,
        height=2,
        text="Reset",
        command=reset_map,
        padx=2,
        pady=2,
        relief=GROOVE)
    reset_button.pack(side=RIGHT)
    frame1.pack(side=BOTTOM)
    create_map(root)
    root.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




