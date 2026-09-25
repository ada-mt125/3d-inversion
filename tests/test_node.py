"""Tests for the DAG engine core."""

import pytest
import numpy as np

from geoinv3d.core.node import Node, Graph, register_node, NODE_REGISTRY


# ── Test fixtures: simple concrete nodes ──────────────────────

class ConstNode(Node[int]):
    def __init__(self, value: int, name: str = "const"):
        super().__init__(name)
        self.value = value

    def _compute(self, inputs):
        return self.value

    def set_value(self, v):
        if v != self.value:
            self.value = v
            self.invalidate()

    def params(self):
        return {"value": self.value, "name": self.name}


class AddNode(Node[int]):
    def __init__(self, a: Node[int], b: Node[int], name: str = "add"):
        super().__init__(name, inputs=[a, b])

    def _compute(self, inputs):
        return inputs[0] + inputs[1]

    def params(self):
        return {"name": self.name}


class DoubleNode(Node[int]):
    def __init__(self, inp: Node[int], name: str = "double"):
        super().__init__(name, inputs=[inp])

    def _compute(self, inputs):
        return inputs[0] * 2


# ── Tests ─────────────────────────────────────────────────────

class TestNode:
    def test_evaluate_basic(self):
        c = ConstNode(5)
        assert c.evaluate() == 5

    def test_evaluate_chain(self):
        a = ConstNode(3)
        b = ConstNode(4)
        s = AddNode(a, b)
        assert s.evaluate() == 7

    def test_caching(self):
        a = ConstNode(10)
        d = DoubleNode(a)
        assert d.evaluate() == 20
        assert d.evaluate() == 20  # cached, no recompute

    def test_invalidation(self):
        a = ConstNode(5)
        d = DoubleNode(a)
        assert d.evaluate() == 10

        a.set_value(7)
        assert d._dirty  # downstream marked dirty
        assert d.evaluate() == 14  # recomputed

    def test_invalidation_chain(self):
        a = ConstNode(2)
        d1 = DoubleNode(a)
        d2 = DoubleNode(d1)
        assert d2.evaluate() == 8  # 2 * 2 * 2

        a.set_value(3)
        assert d1._dirty
        assert d2._dirty
        assert d2.evaluate() == 12  # 3 * 2 * 2

    def test_no_op_change(self):
        a = ConstNode(5)
        d = DoubleNode(a)
        d.evaluate()
        assert not d._dirty

        a.set_value(5)  # same value — no invalidation
        assert not d._dirty

    def test_dependents_tracked(self):
        a = ConstNode(1)
        d = DoubleNode(a)
        assert d in a.dependents

    def test_inputs_property(self):
        a = ConstNode(1)
        b = ConstNode(2)
        s = AddNode(a, b)
        assert len(s.inputs) == 2


class TestGraph:
    def test_add_and_count(self):
        g = Graph()
        a = ConstNode(1)
        b = ConstNode(2)
        s = AddNode(a, b)
        g.add(s)
        assert len(g) == 3  # a, b, s all registered

    def test_remove_leaf(self):
        g = Graph()
        a = ConstNode(1)
        d = DoubleNode(a)
        g.add(d)
        assert len(g) == 2

        g.remove(d)
        assert len(g) == 1

    def test_remove_non_leaf_raises(self):
        g = Graph()
        a = ConstNode(1)
        d = DoubleNode(a)
        g.add(d)

        with pytest.raises(ValueError, match="has dependents"):
            g.remove(a)

    def test_lineage(self):
        g = Graph()
        a = ConstNode(1)
        b = ConstNode(2)
        s = AddNode(a, b)
        d = DoubleNode(s)
        g.add(d)

        lineage = g.lineage(d)
        ids = {n.id for n in lineage}
        assert a.id in ids
        assert b.id in ids
        assert s.id in ids
        assert d.id in ids

    def test_closure(self):
        a = ConstNode(1)
        d1 = DoubleNode(a)
        d2 = DoubleNode(d1)

        closure = Graph.closure(a)
        ids = {n.id for n in closure}
        assert a.id in ids
        assert d1.id in ids
        assert d2.id in ids

    def test_remove_cascade(self):
        g = Graph()
        a = ConstNode(1)
        d1 = DoubleNode(a)
        d2 = DoubleNode(d1)
        g.add(d2)
        assert len(g) == 3

        removed = g.remove_cascade(d1)
        assert len(g) == 1  # only 'a' remains
        removed_ids = {n.id for n in removed}
        assert d1.id in removed_ids
        assert d2.id in removed_ids


class TestNodeParams:
    def test_const_params(self):
        c = ConstNode(42, name="test")
        p = c.params()
        assert p["value"] == 42
        assert p["name"] == "test"
