from dataclasses import dataclass
from typing import Dict, List
import random


WORLD_ID = {
    "number_line": 0,
    "modulo_10": 1,
}


ACTION_TO_ID = {
    -3: 0,
    -2: 1,
    -1: 2,
     1: 3,
     2: 4,
     3: 5,
}

ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}


@dataclass(frozen=True)
class Transition:
    world_name: str
    world_id: int
    state: int
    action: int
    action_id: int
    target: int


class NumberLineWorld:
    name = "number_line"

    def __init__(self, min_number: int = 0, max_number: int = 100):
        self.min_number = min_number
        self.max_number = max_number

    def step(self, state: int, action: int) -> int:
        return state + action

    def valid(self, state: int, action: int) -> bool:
        target = state + action
        return self.min_number <= state <= self.max_number and self.min_number <= target <= self.max_number

    def sample(self, actions: List[int], train_max_number: int) -> Transition:
        while True:
            state = random.randint(self.min_number, train_max_number)
            action = random.choice(actions)
            if self.valid(state, action):
                target = self.step(state, action)
                return Transition(self.name, WORLD_ID[self.name], state, action, ACTION_TO_ID[action], target)


class ModuloWorld:
    def __init__(self, modulo_n: int = 10):
        self.modulo_n = modulo_n
        self.name = f"modulo_{modulo_n}"

    def step(self, state: int, action: int) -> int:
        return (state + action) % self.modulo_n

    def valid(self, state: int, action: int) -> bool:
        return 0 <= state < self.modulo_n

    def sample(self, actions: List[int], train_max_number: int) -> Transition:
        del train_max_number
        state = random.randint(0, self.modulo_n - 1)
        action = random.choice(actions)
        target = self.step(state, action)
        return Transition("modulo_10", WORLD_ID["modulo_10"], state, action, ACTION_TO_ID[action], target)


class MixedWorldSampler:
    def __init__(self, world_names: List[str], max_number: int, train_max_number: int, actions: List[int], modulo_n: int):
        self.actions = actions
        self.train_max_number = train_max_number
        self.worlds = []
        for name in world_names:
            if name == "number_line":
                self.worlds.append(NumberLineWorld(0, max_number))
            elif name == "modulo_10":
                self.worlds.append(ModuloWorld(modulo_n))
            else:
                raise ValueError(f"Unknown world: {name}")

    def sample_batch(self, batch_size: int) -> Dict[str, List[int]]:
        transitions = [random.choice(self.worlds).sample(self.actions, self.train_max_number) for _ in range(batch_size)]
        return {
            "world_id": [t.world_id for t in transitions],
            "state": [t.state for t in transitions],
            "action_id": [t.action_id for t in transitions],
            "target": [t.target for t in transitions],
            "action_value": [t.action for t in transitions],
            "world_name": [t.world_name for t in transitions],
        }


def fixed_probe_set() -> List[Transition]:
    return [
        Transition("number_line", WORLD_ID["number_line"], 3, 2, ACTION_TO_ID[2], 5),
        Transition("number_line", WORLD_ID["number_line"], 50, -3, ACTION_TO_ID[-3], 47),
        Transition("number_line", WORLD_ID["number_line"], 59, 3, ACTION_TO_ID[3], 62),
        Transition("number_line", WORLD_ID["number_line"], 90, 3, ACTION_TO_ID[3], 93),
        Transition("modulo_10", WORLD_ID["modulo_10"], 8, 3, ACTION_TO_ID[3], 1),
        Transition("modulo_10", WORLD_ID["modulo_10"], 9, 1, ACTION_TO_ID[1], 0),
        Transition("modulo_10", WORLD_ID["modulo_10"], 0, -1, ACTION_TO_ID[-1], 9),
        Transition("modulo_10", WORLD_ID["modulo_10"], 4, 3, ACTION_TO_ID[3], 7),
    ]
