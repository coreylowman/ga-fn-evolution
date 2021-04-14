from typing import List, Dict
from copy import deepcopy
import numpy as np
import gym
import atexit
import networkx as nx


def relu(x: np.ndarray):
    return np.maximum(x, 0)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def square(x):
    return np.power(x, 2)


def identity(x):
    return x


class Constant:
    def __init__(self, id: int, value: float):
        self.id = id
        self.value = value

    def __repr__(self):
        return f"Constant({self.id} {self.value})"


class Node:
    def __init__(self, id: int):
        self.id = id
        self.value = None

    def __call__(self, *args, **kwargs):
        return self.value


class ComputationNode(Node):
    AGG_FNS = {np.sum, np.prod, np.amin, np.amax}
    ACT_FNS = {relu, np.tanh, np.sin, np.cos, sigmoid, square, np.sqrt, identity}

    def __init__(self, node_id: int, agg_fn, act_fn, bias_var: int, sign: float):
        assert agg_fn in self.AGG_FNS
        assert act_fn in self.ACT_FNS

        super().__init__(node_id)

        self.agg_fn = agg_fn
        self.act_fn = act_fn
        self.sign = sign
        self.bias_var = bias_var

    def __call__(self, x: np.ndarray, ctx: Dict[int, float]):
        if self.value is None:
            x = self.agg_fn(x, axis=-1)
            assert self.sign == 1.0 or self.sign == -1.0
            x += self.sign * ctx[self.bias_var]
            x = self.act_fn(x)
            self.value = x
        return self.value

    def __repr__(self):
        return f"Node({self.id}, {self.act_fn.__name__}({self.agg_fn.__name__}(x) + {int(self.sign)} * V({self.bias_var})))"


class Edge:
    def __init__(
        self, id: int, from_node_id: int, to_node_id: int, weight_var: int, sign: float
    ):
        self.id = id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.weight_var = weight_var
        self.sign = sign

    def __call__(self, x: np.ndarray, ctx: Dict[int, float]):
        assert self.sign == 1.0 or self.sign == -1.0
        return self.sign * ctx[self.weight_var] * x

    def __repr__(self):
        return f"Edge({self.id}, {self.from_node_id}->{self.to_node_id}, w={int(self.sign)} * V({self.weight_var}))"


class ComputationGraph:
    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        hidden_nodes: List[ComputationNode],
        output_nodes: List[ComputationNode],
        edges: List[Edge],
        variables: List[Constant],
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

        self.variables = variables
        self.ctx = {v.id: v.value for v in self.variables}

        self.nodes = self.inp_nodes + self.out_nodes + self.hidden_nodes
        self.node_by_id = {node.id: node for node in self.nodes}

        self.edges = edges
        self.edge_by_id = {edge.id: edge for edge in self.edges}

        self.edges_by_from = {id: set() for id in self.node_by_id.keys()}
        self.edges_by_to = {id: set() for id in self.node_by_id.keys()}
        for edge in self.edges:
            self.edges_by_from[edge.from_node_id].add(edge.id)
            self.edges_by_to[edge.to_node_id].add(edge.id)

        self.g = nx.DiGraph()
        for id in self.node_by_id:
            self.g.add_node(id)
        for edge in self.edges:
            self.g.add_edge(edge.from_node_id, edge.to_node_id)

    def __repr__(self):
        parts = (
            list(map(str, self.hidden_nodes))
            + list(map(str, self.out_nodes))
            + list(map(str, self.edges))
            + list(map(str, self.variables))
        )
        return "\n".join(parts)

    def num_unlinked_output_nodes(self):
        return sum(
            [1 for node in self.out_nodes if len(self.edges_by_to[node.id]) == 0]
        )

    def __call__(self, x: np.ndarray):
        batch_size = x.shape[0]

        for node in self.nodes:
            node.value = None

        parents_by_node = deepcopy(self.edges_by_to)

        # set input node values
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
                    inp[:, i] = edge(self.node_by_id[edge.from_node_id].value, self.ctx)

                # call node
                self.node_by_id[to_node_id](inp, self.ctx)

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
                out[:, i] = np.zeros((batch_size,))
            else:
                assert node.value is not None
                assert node.value.shape[0] == batch_size
                out[:, i] = node.value

        for node in self.nodes:
            node.value = None

        return out


def candidate_acyclic_edges(graph: ComputationGraph):
    edges = []
    g = graph.g
    for from_node in graph.nodes:
        preds = nx.ancestors(g, from_node.id)
        for to_node in graph.hidden_nodes + graph.out_nodes:
            if from_node.id == to_node.id or to_node.id in preds:
                continue
            edges.append((from_node.id, to_node.id))
    return edges


def candidate_acyclic_edges_from(graph: ComputationGraph, edge: Edge):
    # keep from the same
    edges = []
    g = graph.g.copy()
    g.remove_edge(edge.from_node_id, edge.to_node_id)
    preds = nx.ancestors(g, edge.from_node_id)
    for to_node in graph.hidden_nodes + graph.out_nodes:
        if edge.from_node_id == to_node.id or to_node.id in preds:
            continue
        edges.append((edge.from_node_id, to_node.id))
    return edges


def candidate_acyclic_edges_to(graph: ComputationGraph, edge: Edge):
    # keep to the same
    edges = []
    g = graph.g.copy()
    g.remove_edge(edge.from_node_id, edge.to_node_id)
    for from_node in graph.nodes:
        preds = nx.ancestors(g, from_node.id)
        if from_node.id == edge.to_node_id or edge.to_node_id in preds:
            continue
        edges.append((from_node.id, edge.to_node_id))
    return edges


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
    variables = deepcopy(graph.variables)
    next_variable_id = 0 if len(variables) == 0 else max(v.id for v in variables) + 1

    assert nx.is_directed_acyclic_graph(graph.g)

    if node_or_edge == 0:
        if structure_or_values == 0 and len(hidden_nodes) > 0:
            # chg node
            node = np_random.choice(hidden_nodes + output_nodes)
            agg_act_bias_sign = np_random.choice([0, 1, 2, 3])
            if agg_act_bias_sign == 0:
                node.agg_fn = np_random.choice(list(node.AGG_FNS - {node.agg_fn}))
            elif agg_act_bias_sign == 1:
                node.act_fn = np_random.choice(list(node.ACT_FNS - {node.act_fn}))
            elif agg_act_bias_sign == 2:
                node.bias_var = np_random.choice(variables).id
            else:
                node.sign *= -1.0
        else:
            add_or_rem = np_random.choice([0, 1])
            if add_or_rem == 0 or len(hidden_nodes) == 0:
                # add node
                # note: not adding edge for now, so it will be disconnected
                new_variable = Constant(next_variable_id, np_random.uniform(0.0, 1.0))
                new_node = random_node(
                    np_random, next_node_id, variables + [new_variable]
                )
                if new_node.bias_var == new_variable.id:
                    variables.append(new_variable)
                hidden_nodes.append(new_node)
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
    elif node_or_edge == 1:
        if structure_or_values == 0 and len(edges) > 0:
            # chg edge weight variable
            edge = np_random.choice(edges)
            weight_sign_from_to = np_random.choice([0, 1, 2, 3])
            if weight_sign_from_to == 0:
                # weight variable
                new_variable = Constant(next_variable_id, np_random.uniform(0.0, 1.0))
                edge.weight_var = np_random.choice(variables + [new_variable]).id
                if edge.weight_var == new_variable.id:
                    variables.append(new_variable)
            elif weight_sign_from_to == 1:
                # sign
                edge.sign *= -1.0
            elif weight_sign_from_to == 2:
                # from
                candidate_edges = candidate_acyclic_edges_to(graph, edge)
                from_node_id, to_node_id = candidate_edges[
                    np_random.choice(list(range(len(candidate_edges))))
                ]
                edge.from_node_id = from_node_id
                assert to_node_id == edge.to_node_id
            elif weight_sign_from_to == 3:
                # to
                candidate_edges = candidate_acyclic_edges_from(graph, edge)
                from_node_id, to_node_id = candidate_edges[
                    np_random.choice(list(range(len(candidate_edges))))
                ]
                edge.to_node_id = to_node_id
                assert from_node_id == edge.from_node_id
        else:
            add_or_rem = np_random.choice([0, 1])
            if add_or_rem == 0 or len(edges) == 0:
                # add edge
                candidate_edges = candidate_acyclic_edges(graph)
                from_node_id, to_node_id = candidate_edges[
                    np_random.choice(list(range(len(candidate_edges))))
                ]
                variable = np_random.choice(variables).id
                sign = np_random.choice([-1.0, 1.0])
                edges.append(
                    Edge(next_edge_id, from_node_id, to_node_id, variable, sign)
                )
            else:
                # rem edge
                edge_ind = np_random.choice(list(range(len(edges))))
                edges.pop(edge_ind)

                # note: doesn't matter if it disconnects an edge

    graph2 = ComputationGraph(
        graph.inp_dim, graph.out_dim, hidden_nodes, output_nodes, edges, variables
    )

    assert nx.is_directed_acyclic_graph(graph2.g)

    return graph2


def random_node(
    np_random: np.random.RandomState, node_id: int, variables: List[Constant]
) -> ComputationNode:
    return ComputationNode(
        node_id,
        np_random.choice(list(ComputationNode.AGG_FNS)),
        np_random.choice(list(ComputationNode.ACT_FNS)),
        np_random.choice(variables).id,
        np_random.choice([1.0, -1.0]),
    )


def random_graph(
    np_random: np.random.RandomState, inp_dim: int, out_dim: int, min_mutations: int
) -> ComputationGraph:
    variables = [
        Constant(0, 0.0),
        Constant(1, 1.0),
        Constant(2, 1 / 2),
        Constant(3, 1 / 3),
        Constant(4, 1 / 4),
        Constant(5, np.pi),
    ]
    graph = ComputationGraph(
        inp_dim,
        out_dim,
        hidden_nodes=[],
        output_nodes=[
            random_node(np_random, inp_dim + i, variables) for i in range(out_dim)
        ],
        edges=[],
        variables=variables,
    )
    for i in range(min_mutations):
        graph = mutate_graph(np_random, graph)
    while np_random.choice([0, 1]) == 0:
        graph = mutate_graph(np_random, graph)
    return graph


def cartpole_handcoded_test():
    graph = ComputationGraph(
        4,
        2,
        hidden_nodes=[],
        output_nodes=[
            ComputationNode(4, np.sum, relu, 0, 1.0),
            ComputationNode(5, np.sum, relu, 0, 1.0),
        ],
        edges=[
            Edge(0, 2, 5, 1, 1.0),
            Edge(1, 3, 5, 2, 1.0),
            Edge(2, 2, 4, 1, -1.0),
            Edge(3, 3, 4, 2, -1.0),
        ],
        variables=[
            Constant(0, 0.0),
            Constant(1, 1.0),
            Constant(2, 0.2),
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


def evolution(population_size=100, elite=3, k=10, mutations=1):
    # env = gym.make("CartPole-v1")
    # env = gym.make("MountainCar-v0")
    env = gym.make("Acrobot-v1")
    print(env.observation_space)
    print(env.action_space)

    seed = 0
    np_random = np.random.RandomState(seed=seed)
    env.seed(seed)
    env.action_space.seed(seed)

    def score(graph: ComputationGraph, render=False) -> float:
        obs = env.reset()
        done = False
        score = 0
        if render:
            env.render()
        while not done:
            out = graph(np.expand_dims(obs, 0))[0]
            obs, reward, done, info = env.step(np.argmax(out))
            if render:
                env.render()
            score += reward
        if render:
            env.render()
        node_penalty = -len(graph.hidden_nodes)
        edge_penalty = -len(graph.edges)
        unlinked_output_penalty = -graph.num_unlinked_output_nodes()
        return score + node_penalty + edge_penalty + unlinked_output_penalty

    population = [
        random_graph(
            np_random,
            env.observation_space.shape[0],
            env.action_space.n,
            min_mutations=5,
        )
        for i in range(population_size)
    ]
    scores = [score(population[i]) for i in range(population_size)]

    def print_best():
        best_i = max(list(range(population_size)), key=lambda i: scores[i])
        print(epoch, max(scores), scores[best_i])
        print(population[best_i])
        print(population[best_i].num_unlinked_output_nodes(), "unlinked")
        print(score(population[best_i], render=True))

    atexit.register(print_best)
    epoch = 0
    while True:
        print(epoch, max(scores), sum(scores) / population_size)
        sorted_inds = sorted(
            list(range(population_size)), key=lambda i: scores[i], reverse=True
        )
        next_population = []
        for i in sorted_inds[:elite]:
            next_population.append(population[i])
            next_population.append(mutate_graph(np_random, population[i]))
        while len(next_population) != population_size:
            tournament_inds = np_random.choice(sorted_inds, replace=False, size=k)
            winner = population[max(tournament_inds, key=lambda i: scores[i])]
            for _ in range(mutations):
                winner = mutate_graph(np_random, winner)
            next_population.append(winner)
        assert len(next_population) == population_size, len(next_population)

        next_scores = [score(next_population[i]) for i in range(population_size)]

        population = next_population
        scores = next_scores

        epoch += 1


def main():
    # cartpole_handcoded_test()
    evolution()


if __name__ == "__main__":
    main()
