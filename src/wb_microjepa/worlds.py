from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
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

    def _sample_state_from_ranges(self, ranges: Sequence[Tuple[int, int]], weights: Optional[Sequence[float]] = None) -> int:
        if not ranges:
            raise ValueError("NumberLineWorld received empty ranges.")
        idx = random.choices(range(len(ranges)), weights=weights if weights is not None else None, k=1)[0]
        start, end = ranges[idx]
        return random.randint(int(start), int(end))

    def _sample_action(self, actions: List[int], action_weights: Optional[Dict[int, float]] = None) -> int:
        if action_weights:
            normalized = {int(k): float(v) for k, v in action_weights.items()}
            weights = [float(normalized.get(action, 0.0)) for action in actions]
            if sum(weights) > 0:
                return random.choices(actions, weights=weights, k=1)[0]
        return random.choice(actions)

    def sample(
        self,
        actions: List[int],
        train_max_number: int,
        train_ranges: Optional[Sequence[Tuple[int, int]]] = None,
        state_ranges: Optional[Sequence[Tuple[int, int]]] = None,
        state_range_weights: Optional[Sequence[float]] = None,
        action_weights: Optional[Dict[int, float]] = None,
    ) -> Transition:
        while True:
            if state_ranges:
                state = self._sample_state_from_ranges(state_ranges, state_range_weights)
            elif train_ranges:
                state = self._sample_state_from_ranges(train_ranges)
            else:
                state = random.randint(self.min_number, train_max_number)
            action = self._sample_action(actions, action_weights)
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

    def _sample_state_from_ranges(self, ranges: Sequence[Tuple[int, int]], weights: Optional[Sequence[float]] = None) -> int:
        if not ranges:
            raise ValueError("ModuloWorld received empty ranges.")
        idx = random.choices(range(len(ranges)), weights=weights if weights is not None else None, k=1)[0]
        start, end = ranges[idx]
        return random.randint(max(0, int(start)), min(self.modulo_n - 1, int(end)))

    def _sample_action(self, actions: List[int], action_weights: Optional[Dict[int, float]] = None) -> int:
        if action_weights:
            normalized = {int(k): float(v) for k, v in action_weights.items()}
            weights = [float(normalized.get(action, 0.0)) for action in actions]
            if sum(weights) > 0:
                return random.choices(actions, weights=weights, k=1)[0]
        return random.choice(actions)

    def sample(
        self,
        actions: List[int],
        train_max_number: int,
        train_ranges: Optional[Sequence[Tuple[int, int]]] = None,
        state_ranges: Optional[Sequence[Tuple[int, int]]] = None,
        state_range_weights: Optional[Sequence[float]] = None,
        action_weights: Optional[Dict[int, float]] = None,
    ) -> Transition:
        del train_max_number
        del train_ranges
        if state_ranges:
            state = self._sample_state_from_ranges(state_ranges, state_range_weights)
        else:
            state = random.randint(0, self.modulo_n - 1)
        action = self._sample_action(actions, action_weights)
        target = self.step(state, action)
        return Transition("modulo_10", WORLD_ID["modulo_10"], state, action, ACTION_TO_ID[action], target)


class MixedWorldSampler:
    def __init__(
        self,
        world_names: List[str],
        max_number: int,
        train_max_number: int,
        actions: List[int],
        modulo_n: int,
        train_ranges: Optional[List[List[int]]] = None,
        curriculum_phases: Optional[List[Dict]] = None,
        curriculum_schedule: Optional[List[Dict]] = None,
    ):
        self.actions = actions
        self.train_max_number = train_max_number
        self.train_ranges = [tuple(r) for r in train_ranges] if train_ranges else None
        self.curriculum_phases = curriculum_phases or []
        self.curriculum_schedule = curriculum_schedule or []
        self.worlds = []
        for name in world_names:
            if name == "number_line":
                self.worlds.append(NumberLineWorld(0, max_number))
            elif name == "modulo_10":
                self.worlds.append(ModuloWorld(modulo_n))
            else:
                raise ValueError(f"Unknown world: {name}")

    def _phase_for_epoch(self, epoch: Optional[int]) -> Optional[Dict]:
        if epoch is None:
            return None
        if self.curriculum_schedule:
            return self._scheduled_phase_for_epoch(epoch)
        if epoch is None or not self.curriculum_phases:
            return None
        for phase in self.curriculum_phases:
            if int(phase.get("start_epoch", 1)) <= epoch <= int(phase.get("end_epoch", epoch)):
                return phase
        return self.curriculum_phases[-1]

    def _blend_values(self, left, right, t: float):
        if left is None:
            return right
        if right is None:
            return left
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return float(left) + (float(right) - float(left)) * t
        if isinstance(left, dict) and isinstance(right, dict):
            keys = set(left) | set(right)
            blended = {}
            for key in keys:
                if key in left and key in right:
                    blended[key] = self._blend_values(left[key], right[key], t)
                elif key in left:
                    blended[key] = left[key]
                else:
                    blended[key] = right[key]
            return blended
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right):
            return [self._blend_values(lv, rv, t) for lv, rv in zip(left, right)]
        return right if t >= 0.5 else left

    def _scheduled_phase_for_epoch(self, epoch: int) -> Dict:
        schedule = sorted(self.curriculum_schedule, key=lambda p: int(p.get("epoch", 1)))
        if not schedule:
            return {}
        if len(schedule) == 1:
            return dict(schedule[0])

        if epoch <= int(schedule[0].get("epoch", 1)):
            return dict(schedule[0])
        if epoch >= int(schedule[-1].get("epoch", epoch)):
            return dict(schedule[-1])

        for left, right in zip(schedule, schedule[1:]):
            left_epoch = int(left.get("epoch", 1))
            right_epoch = int(right.get("epoch", left_epoch))
            if left_epoch <= epoch <= right_epoch:
                if right_epoch == left_epoch:
                    return dict(left)
                t = (epoch - left_epoch) / float(right_epoch - left_epoch)
                blended = {}
                keys = set(left) | set(right)
                for key in keys:
                    if key == "epoch":
                        blended[key] = epoch
                    elif key == "name":
                        left_name = str(left.get("name", "phase"))
                        right_name = str(right.get("name", "phase"))
                        blended[key] = f"{left_name}_to_{right_name}"
                    else:
                        blended[key] = self._blend_values(left.get(key), right.get(key), t)
                return blended

        return dict(schedule[-1])

    def current_phase_name(self, epoch: Optional[int]) -> str:
        phase = self._phase_for_epoch(epoch)
        return str(phase.get("name", "default")) if phase else "default"

    def sample_batch(self, batch_size: int, epoch: Optional[int] = None) -> Dict[str, List[int]]:
        phase = self._phase_for_epoch(epoch)
        transitions = [
            self._sample_transition(phase)
            for _ in range(batch_size)
        ]
        return {
            "world_id": [t.world_id for t in transitions],
            "state": [t.state for t in transitions],
            "action_id": [t.action_id for t in transitions],
            "target": [t.target for t in transitions],
            "action_value": [t.action for t in transitions],
            "world_name": [t.world_name for t in transitions],
            "curriculum_phase": [self.current_phase_name(epoch)] * batch_size,
        }

    def _sample_transition(self, phase: Optional[Dict]) -> Transition:
        phase = phase or {}
        world_weights = phase.get("world_weights")
        if world_weights:
            worlds = [w for w in self.worlds if w.name in world_weights and float(world_weights.get(w.name, 0.0)) > 0.0]
            if not worlds:
                worlds = self.worlds
            selected_world = random.choices(worlds, weights=[float(world_weights.get(w.name, 0.0)) for w in worlds], k=1)[0]
        else:
            selected_world = random.choice(self.worlds)

        number_line_ranges = phase.get("number_line_ranges")
        number_line_range_weights = phase.get("number_line_range_weights")
        modulo_state_ranges = phase.get("modulo_state_ranges")
        modulo_state_range_weights = phase.get("modulo_state_range_weights")
        action_weights = phase.get("action_weights")

        if selected_world.name == "number_line":
            return selected_world.sample(
                self.actions,
                self.train_max_number,
                self.train_ranges,
                state_ranges=[tuple(r) for r in number_line_ranges] if number_line_ranges else self.train_ranges,
                state_range_weights=number_line_range_weights,
                action_weights=action_weights,
            )
        if selected_world.name == "modulo_10":
            return selected_world.sample(
                self.actions,
                self.train_max_number,
                self.train_ranges,
                state_ranges=[tuple(r) for r in modulo_state_ranges] if modulo_state_ranges else None,
                state_range_weights=modulo_state_range_weights,
                action_weights=action_weights,
            )
        raise ValueError(f"Unknown world selected: {selected_world.name}")


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
