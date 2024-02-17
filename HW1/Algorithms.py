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


    def pop_back(self):
        if len(self.stack) == 0:
            return None
        return self.stack.pop(0)

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

    def is_in_close(self, item) -> bool:
        temp_list = list(self.stack)
        if len(temp_list) == 0:
            return False
        if item in temp_list:
            return True
        return False

    def is_in_open(self, item) -> bool:
        temp_list = list(self.stack)
        if len(temp_list) == 0:
            return False
        for e in temp_list:
            if e.get_state() == item.get_state():
                return True
        return False


class Node:
    def __init__(self, state, father):
        self.state = state
        self.father = father
        self.cost = 1

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_father(self):
        return self.father

    def update_dragon_ball(self, env):
        location, d1, d2 = self.get_state()
        location_f, d1_f, d2_f = self.get_father().get_state()
        location_d1, d1_d1, d2_d1 = env.d1
        location_d2, d1_d2, d2_d2 = env.d2
        if location == location_d1 and not d1_f:
            d1_f = True
        if location == location_d2 and not d2_f:
            d2_f = True
        new_state = (location, d1_f, d2_f)
        self.set_state(new_state)

    def set_cost(self, cost):
        self.cost = cost

    def get_cost(self):
        return self.cost


def correct_path(actions_lst):
    for i, e in enumerate(actions):
        if e == 2:
            actions_lst[i] = 0
        elif e == 0:
            actions_lst[i] = 2
        elif e == 1:
            actions_lst[i] = 3
        elif e == 3:
            actions_lst[i] = 1

class BFSAgent():
    def __init__(self) -> None:
        self.env = None
        self.actions = Stack()

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        OPEN = Stack()
        CLOSE = Stack()
        node = Node(self.env.get_state(), None)
        if env.is_final_state(node.state):
            return self.solution(node)
        OPEN.push(node)
        while not OPEN.is_empty():
            node = OPEN.pop_back()
            self.env.set_state(node.get_state())
            if node.father != None:
                node.update_dragon_ball(self.env)
            CLOSE.push(node.get_state())
            for action, (state, cost, terminated) in env.succ(node.get_state()).items():
                child = Node(state, node)
                child.update_dragon_ball(self.env)
                if self.is_hole(child, terminated):
                    continue
                if not CLOSE.is_in_close(child.get_state()) and not OPEN.is_in_open(child):
                    if self.env.is_final_state(child.get_state()):
                        if self.env.is_final_state(child.get_state()):
                            return self.solution(child)
                    child.set_cost(cost)
                    OPEN.push(child)

    def is_hole(self, node: Node, terminate) -> bool:
        state = node.get_state()
        if terminate and not self.env.is_final_state(state):
            return True
        return False


    @staticmethod
    def solution(node):
        actions_lst = []
        total_cost = 0
        expanded = 0
        while node.father is not None:
            father_node = node.father
            delta = father_node.get_state()[0] - node.get_state()[0]
            if delta == 8:
                actions_lst.append(Down)
            elif delta == 1:
                actions_lst.append(Right)
            elif delta == -1:
                actions_lst.append(Left)
            elif delta == -8:
                actions_lst.append(Up)
            expanded = expanded + 1
            total_cost = total_cost + father_node.get_cost()
            node = node.father
        return actions_lst, total_cost, expanded


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


MAPS = {
    "4x4": ["SFFF",
            "FDFF",
            "FFFD",
            "FFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFTAL",
        "TFFHFFTF",
        "FFFFFHTF",
        "FAFHFFFF",
        "FHHFFFHF",
        "DFTFHDTL",
        "FLFHFFFG",
    ],
}
env = DragonBallEnv(MAPS["8x8"])
BFS_agent = BFSAgent()
actions, t_cost, expanded = BFS_agent.search(env)
actions.reverse()
correct_path(actions)
print(f"Total_cost: {t_cost}")
print(f"Expanded: {expanded}")
print(f"Actions: {actions}")

assert t_cost == 119.0, "Error in total cost returned"
