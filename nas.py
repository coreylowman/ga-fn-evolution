from typing import List
from copy import deepcopy
import numpy as np
import gym
import atexit


def relu(x: np.ndarray):
    return np.maximum(x, 0)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def square(x):
    return np.power(x, 2)


def identity(x):
    return x


class Variable:
    def __init__(self, id: int, value: float):
        self.id = id
        self.value = value


class Node:
    def __init__(self, id: int):
        self.id = id
        self.value = None

    def __call__(self, *args, **kwargs):
        return self.value


class ComputationNode(Node):
    AGG_FNS = {np.sum, np.prod, np.amin, np.amax}
    ACT_FNS = {relu, np.tanh, np.sin, np.cos, sigmoid, square, np.sqrt, identity}

    def __init__(self, node_id: int, agg_fn, act_fn, bias: float):
        assert agg_fn in self.AGG_FNS
        assert act_fn in self.ACT_FNS

        super().__init__(node_id)

        self.agg_fn = agg_fn
        self.act_fn = act_fn
        self.bias = bias

    def __call__(self, x: np.ndarray):
        if self.value is None:
            x = self.agg_fn(x, axis=-1)
            x += self.bias
            x = self.act_fn(x)
            self.value = x
        return self.value

    def __repr__(self):
        return f"Node({self.id}, {self.act_fn.__name__}({self.agg_fn.__name__}(x) + {self.bias}))"


class Edge:
    def __init__(self, id: int, from_node_id: int, to_node_id: int, weight: float):
        self.id = id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.weight = weight

    def __call__(self, x: np.ndarray):
        return self.weight * x

    def __repr__(self):
        return (
            f"Edge({self.id}, {self.from_node_id}->{self.to_node_id}, w={self.weight})"
        )


class ComputationGraph:
    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        hidden_nodes: List[ComputationNode],
        output_nodes: List[ComputationNode],
        edges: List[Edge],
    ):
        assert out_dim == len(output_nodes)
        for i, node in enumerate(output_nodes):
            assert node.id == inp_dim + i

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.node_id = 0
        self.inp_nodes = [Node(self.node_id + i) for i in range(inp_dim)]
        self.node_id += inp_dim
        self.out_nodes = output_nodes
        self.node_id += out_dim
        self.hidden_nodes = hidden_nodes

        self.nodes = self.inp_nodes + self.out_nodes + self.hidden_nodes
        self.node_by_id = {node.id: node for node in self.nodes}

        self.edges = edges
        self.edge_by_id = {edge.id: edge for edge in self.edges}

        self.edges_by_from = {id: set() for id in self.node_by_id.keys()}
        self.edges_by_to = {id: set() for id in self.node_by_id.keys()}
        for edge in self.edges:
            self.edges_by_from[edge.from_node_id].add(edge.id)
            self.edges_by_to[edge.to_node_id].add(edge.id)

    def __repr__(self):
        parts = (
            list(map(str, self.hidden_nodes))
            + list(map(str, self.out_nodes))
            + list(map(str, self.edges))
        )
        return "\n".join(parts)

    def bfs(self):
        layers = [{node.id for node in self.inp_nodes}]

        disconnected = set(self.node_by_id.keys())
        disconnected -= layers[0]

        parents_by_node = deepcopy(self.edges_by_to)

        frontier = set()
        for node in self.inp_nodes:
            for edge_id in self.edges_by_from[node.id]:
                edge = self.edge_by_id[edge_id]
                parents_by_node[edge.to_node_id].discard(edge_id)
                if len(parents_by_node[edge.to_node_id]) == 0:
                    frontier.add(edge.to_node_id)

        while len(frontier) > 0:
            disconnected -= frontier
            layers.append(frontier)
            new_frontier = set()
            for to_node_id in frontier:
                # extend frontier
                for edge_id in self.edges_by_from[to_node_id]:
                    edge = self.edge_by_id[edge_id]
                    parents_by_node[edge.to_node_id].discard(edge_id)
                    if len(parents_by_node[edge.to_node_id]) == 0:
                        new_frontier.add(edge.to_node_id)
            frontier = new_frontier

        assert len(frontier) == 0
        assert sum([len(f) for f in layers]) + len(disconnected) == len(self.nodes)
        if len(disconnected) > 0:
            layers.append(disconnected)
        return layers

    def __call__(self, x: np.ndarray):
        batch_size = x.shape[0]

        for node in self.nodes:
            node.value = None

        parents_by_node = deepcopy(self.edges_by_to)

        frontier = set()
        for node in self.inp_nodes:
            node.value = x[:, node.id]
            for edge_id in self.edges_by_from[node.id]:
                edge = self.edge_by_id[edge_id]
                parents_by_node[edge.to_node_id].discard(edge_id)
                if len(parents_by_node[edge.to_node_id]) == 0:
                    frontier.add(edge.to_node_id)

        while len(frontier) > 0:
            new_frontier = set()
            for to_node_id in frontier:
                inp = np.zeros((batch_size, len(self.edges_by_to[to_node_id])))

                # collect input
                for i, edge_id in enumerate(self.edges_by_to[to_node_id]):
                    edge = self.edge_by_id[edge_id]
                    inp[:, i] = edge(self.node_by_id[edge.from_node_id].value)

                # call node
                self.node_by_id[to_node_id](inp)

                # extend frontier
                for edge_id in self.edges_by_from[to_node_id]:
                    edge = self.edge_by_id[edge_id]
                    parents_by_node[edge.to_node_id].discard(edge_id)
                    if len(parents_by_node[edge.to_node_id]) == 0:
                        new_frontier.add(edge.to_node_id)
            frontier = new_frontier

        # collect output values
        out = np.zeros((batch_size, self.out_dim))
        for i, node in enumerate(self.out_nodes):
            if node.value is None:
                out[:, i] = np.random.uniform(-1.0, 1.0, size=(batch_size,))
            else:
                assert node.value is not None
                assert node.value.shape[0] == batch_size
                out[:, i] = node.value

        for node in self.nodes:
            node.value = None

        return out


def mutate_graph(
    np_random: np.random.RandomState, graph: ComputationGraph
) -> ComputationGraph:
    node_or_edge = np_random.choice([0, 1])
    structure_or_values = np_random.choice([0, 1])

    hidden_nodes = deepcopy(graph.hidden_nodes)
    output_nodes = deepcopy(graph.out_nodes)
    next_node_id = (
        graph.inp_dim + graph.out_dim
        if len(hidden_nodes) == 0
        else max(node.id for node in hidden_nodes) + 1
    )
    edges = deepcopy(graph.edges)
    next_edge_id = 0 if len(edges) == 0 else max(edge.id for edge in edges) + 1

    layers = graph.bfs()

    if node_or_edge == 0:
        if structure_or_values == 0 and len(hidden_nodes) > 0:
            # chg node
            node = np_random.choice(hidden_nodes + output_nodes)
            agg_act_bias = np_random.choice([0, 1, 2])
            if agg_act_bias == 0:
                node.agg_fn = np_random.choice(list(node.AGG_FNS - {node.agg_fn}))
            elif agg_act_bias == 1:
                node.act_fn = np_random.choice(list(node.ACT_FNS - {node.act_fn}))
            else:
                node.bias = np.clip(node.bias + np_random.normal(scale=0.1), -1.0, 1.0)
        else:
            add_or_rem = np_random.choice([0, 1])
            if add_or_rem == 0 or len(hidden_nodes) == 0:
                # add node
                # note: not adding edge for now, so it will be disconnected
                hidden_nodes.append(random_node(np_random, next_node_id))
            else:
                # rem node (and related edges)
                ind = np_random.choice(list(range(len(hidden_nodes))))
                node = hidden_nodes.pop(ind)

                to_remove = []
                for edge in edges:
                    if edge.from_node_id == node.id or edge.to_node_id == node.id:
                        to_remove.append(edge)
                for edge in to_remove:
                    edges.remove(edge)
    else:
        if structure_or_values == 0 and len(edges) > 0:
            # chg edge weight
            edge = np_random.choice(edges)
            edge.weight = np_random.uniform(-1.0, 1.0)
        else:
            add_or_rem = np_random.choice([0, 1])
            if add_or_rem == 0 or len(edges) == 0:
                # add edge
                nodes_in_front = [item for sublist in layers[:-1] for item in sublist]
                from_node_id = np_random.choice(nodes_in_front)
                for layer_id, layer in enumerate(layers):
                    if from_node_id in layer:
                        break
                nodes_after = [
                    item for sublist in layers[layer_id + 1 :] for item in sublist
                ]
                to_node_id = np_random.choice(nodes_after)
                weight = np_random.uniform(-1.0, 1.0)
                edges.append(Edge(next_edge_id, from_node_id, to_node_id, weight))

            else:
                # rem edge
                edge_ind = np_random.choice(list(range(len(edges))))
                edges.pop(edge_ind)

                # note: doesn't matter if it disconnects an edge

    return ComputationGraph(
        graph.inp_dim, graph.out_dim, hidden_nodes, output_nodes, edges
    )


def random_node(np_random: np.random.RandomState, node_id: int) -> ComputationNode:
    return ComputationNode(
        node_id,
        np_random.choice(list(ComputationNode.AGG_FNS)),
        np_random.choice(list(ComputationNode.ACT_FNS)),
        np_random.uniform(-1.0, 1.0),
    )


def random_graph(
    np_random: np.random.RandomState, inp_dim: int, out_dim: int, min_mutations: int
) -> ComputationGraph:
    graph = ComputationGraph(
        inp_dim,
        out_dim,
        hidden_nodes=[],
        output_nodes=[random_node(np_random, inp_dim + i) for i in range(out_dim)],
        edges=[],
    )
    for i in range(min_mutations):
        graph = mutate_graph(np_random, graph)
    while np_random.choice([0, 1]) == 0:
        graph = mutate_graph(np_random, graph)
    return graph


def score(graph: ComputationGraph) -> float:
    env = gym.make("CartPole-v1")
    obs = env.reset()
    done = False
    score = 0
    while not done:
        out = graph(np.expand_dims(obs, 0))[0]
        obs, reward, done, info = env.step(np.argmax(out))
        score += reward
    node_penalty = -len(graph.hidden_nodes)
    edge_penalty = -len(graph.edges)
    return score + node_penalty + edge_penalty


def cartpole_handcoded_test():
    graph = ComputationGraph(
        4,
        2,
        hidden_nodes=[],
        output_nodes=[
            ComputationNode(4, np.sum, relu, 0),
            ComputationNode(5, np.sum, relu, 0),
        ],
        edges=[
            Edge(0, 2, 5, 1.0),
            Edge(1, 3, 5, 0.2),
            Edge(2, 2, 4, -1.0),
            Edge(3, 3, 4, -0.2),
        ],
    )

    env = gym.make("CartPole-v1")
    env.seed(0)
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action = 1 if obs[2] + 0.2 * obs[3] > 0 else 0
        out = graph(np.expand_dims(obs, 0))[0]
        print(action, np.argmax(out), out)
        assert np.argmax(out) == action
        obs, reward, done, info = env.step(action)
        score += reward
    print(score)


def evolution(population_size=50, k=5):
    np_random = np.random.RandomState(seed=0)
    population = [
        random_graph(np_random, 4, 2, min_mutations=5) for i in range(population_size)
    ]

    def print_best():
        best_i = max(list(range(population_size)), key=lambda i: scores[i])
        print(epoch, max(scores), scores[best_i])
        print(population[best_i])

    atexit.register(print_best)
    epoch = 0
    while True:
        scores = [score(population[i]) for i in range(population_size)]
        print(epoch, max(scores), scores)
        sorted_inds = sorted(
            list(range(population_size)), key=lambda i: scores[i], reverse=True
        )
        top_k = sorted_inds[:k]
        next_population = []
        for i in top_k:
            next_population.append(population[i])
            for j in range(population_size // k - 1):
                next_population.append(mutate_graph(np_random, population[i]))
        assert len(next_population) == population_size, len(next_population)
        population = next_population
        epoch += 1


def main():
    # cartpole_handcoded_test()
    evolution()


if __name__ == "__main__":
    main()
