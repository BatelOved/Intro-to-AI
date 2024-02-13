import numpy as np
from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

Down = 0
Right = 1
Up = 2
Left = 3


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if len(self.stack) == 0:
            return None
        return self.stack.pop()

    def peek(self):
        if len(self.stack) == 0:
            return None
        return self.stack[-1]

    def is_empty(self) -> bool:
        if len(self.stack) == 0:
            return True
        return False

    def size(self) -> int:
        return len(self.stack)


class Node:
    def __init__(self, state, father):
        self.state = state
        self.father = father

    def get_state(self):
        return self.state


class BFSAgent():
    def __init__(self) -> None:
        self.env = None
        self.actions = []

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        OPEN = []
        CLOSE = []
        total_cast = 1
        expended = 0
        node = Node(self.env.get_state(), None)
        if env.is_final_state(node.state):
            return self.solution(node), total_cast, expended
        OPEN.append(node)
        while len(OPEN) != 0:
            node = OPEN.pop()
            CLOSE.append(node.get_state())
            total_cast += node.get_state()[1]
            expended += 1
            for action, successor in env.succ(node.get_state()).items():
                child = Node(successor, node)
                if child.get_state() not in CLOSE and child not in OPEN:
                    if self.env.is_final_state(child.get_state()):
                        return self.solution(child), total_cast, expended
                    OPEN.append(child)

            # for action, (state, cost, terminated) in env.succ((19, False, False)).items():

    @staticmethod
    def solution(node):
        actions = []
        while node.father is not None:
            father_node = node.father
            delta = father_node.get_state()[0] - node.get_state()[0]
            if delta == 8:
                actions.append(Down)
            elif delta == 1:
                actions.append(Right)
            elif delta == -1:
                actions.append(Left)
            elif delta == -8:
                actions.append(Up)
            node = node.father
        return actions


class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError


class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError