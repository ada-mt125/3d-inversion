"""DAG core: nodes with cached evaluation and dirty propagation.

Adapted from jif3d_visualization's graph engine. Design goals:

* The graph is the system of record — a node's output is *only* a function of
  its parameters and its inputs' outputs.
* Eager evaluation with memoization: evaluate() computes once and caches;
  changing a parameter invalidates the downstream subtree.
* Deterministic: given the same inputs and parameters, _compute must return
  equivalent output.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Iterable, Iterator, Optional, TypeVar

_id_counter = itertools.count(1)

_UNSET: Any = object()

T = TypeVar("T")

InputSlot = tuple[str, "type | tuple[type, ...]"]

NODE_REGISTRY: dict[str, type[Node]] = {}

_NodeType = TypeVar("_NodeType", bound="type[Node]")


def register_node(cls: _NodeType) -> _NodeType:
    """Decorator: make a node type deserializable from saved workflows."""
    held = NODE_REGISTRY.get(cls.__name__)
    if held is not None and held is not cls:
        raise TypeError(
            f"node type {cls.__name__} already registered from {held.__module__}"
        )
    NODE_REGISTRY[cls.__name__] = cls
    return cls


class Node(ABC, Generic[T]):
    """Base class for all DAG nodes.

    Subclass contract:
      _compute(inputs)   — pure function, returns output from evaluated inputs
      params()           — JSON-serializable parameters (no inputs)
      from_params(p, i)  — inverse of params; reconstruct node
    """

    evictable: ClassVar[bool] = True
    reads_file: ClassVar[bool] = False

    def __init__(self, name: str, inputs: Iterable[Node] = ()) -> None:
        self.id: int = next(_id_counter)
        self.name = name
        self._inputs: list[Node] = list(inputs)
        self._dependents: set[Node] = set()
        self._output: Any = _UNSET
        self._dirty: bool = True
        self._generation = 0
        for inp in self._inputs:
            inp._dependents.add(self)

    @property
    def inputs(self) -> list[Node]:
        return list(self._inputs)

    @property
    def dependents(self) -> set[Node]:
        return set(self._dependents)

    @abstractmethod
    def _compute(self, inputs: list[Any]) -> T:
        """Produce this node's output from its evaluated inputs."""

    def input_types(self) -> Optional[tuple[InputSlot, ...]]:
        return None

    def _stale(self) -> bool:
        return self._dirty or self._output is _UNSET

    def evaluate(self) -> T:
        """Return this node's output, computing and caching if dirty.

        Iterative post-order walk (not recursive) to avoid stack overflow
        on long chains.
        """
        if not self._stale():
            return self._output

        stack: list[tuple[Node, bool]] = [(self, False)]
        while stack:
            node, inputs_ready = stack.pop()
            if inputs_ready:
                values = [_resolved(inp) for inp in node._inputs]
                _check_inputs(node, values)
                node._output = node._compute(values)
                node._dirty = False
            elif node._stale():
                stack.append((node, True))
                stack.extend(
                    (inp, False)
                    for inp in reversed(node._inputs)
                    if isinstance(inp, Node) and inp._stale()
                )
        return self._output

    def invalidate(self) -> None:
        """Mark this node and its whole downstream subtree dirty."""
        self._generation += 1
        stack: list[Node] = [self]
        while stack:
            node = stack.pop()
            if node._dirty and node is not self:
                continue
            node._dirty = True
            if node is not self:
                node._generation += 1
            stack.extend(node._dependents)

    def params(self) -> dict:
        """This node's own serializable parameters."""
        return {}

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> Node:
        raise NotImplementedError(f"{cls.__name__} is not deserializable")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} #{self.id} {self.name!r}>"


class Graph:
    """Container of nodes with traversal and provenance helpers."""

    def __init__(self) -> None:
        self._nodes: dict[int, Node] = {}

    def add(self, node: Node) -> Node:
        """Register node and its transitive inputs."""
        stack = [node]
        while stack:
            n = stack.pop()
            if n.id not in self._nodes:
                self._nodes[n.id] = n
                stack.extend(n.inputs)
        return node

    def remove(self, node: Node) -> None:
        """Remove a leaf node (one with no dependents)."""
        if node._dependents:
            raise ValueError(f"cannot remove node #{node.id} — has dependents")
        for inp in node._inputs:
            inp._dependents.discard(node)
        self._nodes.pop(node.id, None)

    def remove_cascade(self, node: Node) -> list[Node]:
        """Remove node and its entire downstream closure."""
        closure = {n.id: n for n in self.closure(node)}
        removed: list[Node] = []
        pending = dict(closure)
        while pending:
            leaves = [n for n in pending.values()
                      if not n._dependents or all(
                          d.id not in pending for d in n._dependents)]
            for leaf in leaves:
                for inp in leaf._inputs:
                    inp._dependents.discard(leaf)
                pending.pop(leaf.id, None)
                self._nodes.pop(leaf.id, None)
                removed.append(leaf)
        return removed

    @staticmethod
    def closure(node: Node) -> list[Node]:
        """node plus every node that transitively depends on it."""
        seen: dict[int, Node] = {}
        stack = [node]
        while stack:
            n = stack.pop()
            if n.id not in seen:
                seen[n.id] = n
                stack.extend(n._dependents)
        return list(seen.values())

    def lineage(self, node: Node) -> list[Node]:
        """All ancestors of node (inputs, transitively) plus node itself."""
        seen: dict[int, Node] = {}
        stack = [node]
        while stack:
            n = stack.pop()
            if n.id not in seen:
                seen[n.id] = n
                stack.extend(n.inputs)
        return list(seen.values())

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes.values())


def _resolved(inp: Any) -> Any:
    if isinstance(inp, Node) and not inp._stale():
        return inp._output
    return inp.evaluate()


def _check_inputs(node: Node, values: list[Any]) -> None:
    declared = node.input_types()
    if declared is None:
        return
    if len(declared) != len(values):
        raise TypeError(
            f"{type(node).__name__} declares {len(declared)} inputs but has {len(values)}"
        )
    for i, ((name, accepted), value) in enumerate(zip(declared, values)):
        if not isinstance(value, accepted):
            wanted = (
                " or ".join(t.__name__ for t in accepted)
                if isinstance(accepted, tuple)
                else accepted.__name__
            )
            raise TypeError(
                f"{type(node).__name__} input {i} ({name}) is "
                f"{type(value).__name__}; expected {wanted}"
            )
