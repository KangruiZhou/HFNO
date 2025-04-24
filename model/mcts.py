import math
import random
import hashlib
from kdtree_mcts import *
import numpy as np
import warnings

class State():
    def __init__(self, value=0, moves=[], turn=6):
        self.value = value
        self.turn = turn
        self.moves = moves

    def next_state(self, Tree, gradient_global_var, cal_reward=False):
        turn = self.turn
        if self.moves and not cal_reward:
            Tree.split_move(self.moves)
        else:
            Tree = Tree
        nextmove_dim, KDode_chosen, split_val = Tree.split()
        nextmove_value = Tree.get_var_sum() / gradient_global_var / Tree.total_num
        next = State(nextmove_value,
                    self.moves + [[nextmove_dim, KDode_chosen, split_val]], turn - 1)
        return next

    def terminal(self):
        if self.turn == 0:
            return True
        return False

    def reward(self):
        r = (1 - self.value)*2
        return r

    def __hash__(self):
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = "Value: %f; Moves: %s" % (self.value, self.moves)
        return s


class Node():
    def __init__(self, state, parent=None):
        self.visits = 0
        self.total_reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def enough_expanded(self):
        if len(self.children) == 200 :
            return True
        return False

    def __repr__(self):
        s = "Node; children: %d; visits: %d; total_reward: %f" % (len(self.children), self.visits, self.total_reward)
        return s


def UCTSEARCH(budget, root, points_indices, gradient_norm_list, gradient_global_var, dim, SCALAR):
    for iter in range(int(budget)):
        Tree = KDTree(points_indices, dim=dim, n_blocks=2 ** dim,
                    gradient_norm_list=gradient_norm_list)
        if iter % 100 == 0:
            print("simulation: %d" % iter)
        front = TREEPOLICY(root, Tree, gradient_global_var, SCALAR)
        reward = DEFAULTPOLICY(front.state, Tree, gradient_global_var)
        BACKUP(front, reward)
    return BESTCHILD(root, 0)


def TREEPOLICY(node, Tree, gradient_global_var, SCALAR):
    # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.state.terminal() == False:
        if len(node.children) == 0:
            return EXPAND(node, Tree, gradient_global_var)
        elif random.uniform(0, 1) < .5:
            node = BESTCHILD(node, SCALAR)
        else:
            if node.enough_expanded() == False:
                return EXPAND(node, Tree, gradient_global_var)
            else:
                node = BESTCHILD(node, SCALAR)
    return node

def EXPAND(node, Tree, gradient_global_var):
    new_state = node.state.next_state(Tree, gradient_global_var)
    node.add_child(new_state)
    return node.children[-1]

# current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node, scalar):
    bestscore = -10000000
    bestchildren = []
    for c in node.children:
        exploit = c.total_reward / c.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
        score = exploit + scalar * explore
        if score == bestscore:
            bestchildren.append(c)
        if score > bestscore:
            bestchildren = [c]
            bestscore = score
    if len(bestchildren) == 0:
        msg = "OOPS: no best child found, probably fatal"
        warnings.warn(msg, category=RuntimeWarning, stacklevel=2)  

    return random.choice(bestchildren)

def DEFAULTPOLICY(state,Tree,gradient_global_var):
    while state.terminal()==False:
        state=state.next_state(Tree, gradient_global_var, cal_reward=True)
    return state.reward()


def BACKUP(node, reward):
    while node != None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent
    return


