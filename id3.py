from dataclasses import dataclass
from typing import Set

import numpy as np
from treelib import Tree


def entropy(target: np.array) -> float:
    """
    Compute the entropy for a target variable, e.g. entropy([1, 0, 1, 0]) is 1.
    """
    _, counts = np.unique(target, return_counts=True)
    proportions = counts / np.sum(counts)
    return np.sum(np.multiply(-proportions, np.log2(proportions)))


def gain(data: np.array, target: np.array, split: int) -> float:
    """
    Given a 2-dim array of features, a 1-dim array of target variables and a split column index
    representing the variable on which to split, compute the information gain of the split.

    :param data: 2-dim array of features [features, samples]
    :param target: 1-dim array of targets
    :param split: index of feature to split on
    :return: information gain of splitting
    """
    uniques, indices = np.unique(data[:, split], return_inverse=True)
    current = entropy(target)
    for value in uniques:
        subset = target[indices == value]
        current -= (len(subset) / len(target)) * entropy(subset)
    return current


@dataclass
class Branch:
    attr: int
    value: int
    data: np.array
    target: np.array
    used: Set[int]


@dataclass
class Leaf:
    label: int


def _id3(data: np.array, target: np.array, used: Set[int], nclasses: np.array):
    """
    Recursive part of the ID3 algorithm. Checks whether any of the stopping criteria have been satisfied, and if
    they have then adds leaf and returns the built tree, otherwise computes the highest gain split and calls
    itself again.

    :param data: 2-dim array of features [features, samples]
    :param target: 1-dim array of targets
    :param used: list of features already used in the splitting higher up the tree
    :param nclasses: number of classes per feature
    :return: built tree
    """
    tree = Tree()
    root = tree.create_node("root")  # redundant nodes we have to later remove
    samples, variables = data.shape
    target_uniques, target_counts = np.unique(target, return_counts=True)
    if len(target_uniques) == 1:
        # all examples belong to a single class, so no more splitting is needed
        tree.create_node(
            tag=f"label = {target_uniques[0]}",
            parent=root,
            data=Leaf(label=target_uniques[0]),
        )
    elif len(used) == variables:
        # all attributes have been used up for splitting, so no more splitting can be done
        label = np.argmax(target_counts)
        tree.create_node(
            tag=f"label = {target_uniques[label]}",
            parent=root,
            data=Leaf(label=target_uniques[label]),
        )
    else:
        gains = np.array(
            [
                gain(data, target, split) if split not in used else -np.inf
                for split in range(variables)
            ]
        )
        split = np.argmax(gains)  # choose the highest gain attributes
        _, indices = np.unique(data[:, split], return_inverse=True)
        for value in range(nclasses[split]):
            mask = indices == value
            data_value = data[mask]
            target_value = target[mask]
            node = tree.create_node(
                tag=f"feature {split} = {value}",
                parent=root,
                data=Branch(
                    attr=split,
                    value=value,
                    data=data_value,
                    target=target_value,
                    used=used.union({split}),
                ),
            )
            if mask.sum() > 0:
                # continue the splitting if there are examples
                tree.paste(
                    node.identifier,
                    _id3(data_value, target_value, used.union({split}), nclasses),
                )
            else:
                # there are no examples, so no more splitting can be done
                label = target_uniques[target_counts.argmax()]
                tree.create_node(
                    tag=f"label = {label}", parent=node, data=Leaf(label=label)
                )
    return tree


def id3(data, target, nclasses: np.array):
    """
    Main part of the ID3 algorithm. Calls the recursive part with the root node, which then builds the tree.

    :param data: 2-dim array of features [features, samples]
    :param target: 1-dim array of targets
    :param nclasses: number of classes per feature
    :return: built tree
    """
    tree = Tree()
    root = tree.create_node("root")
    tree.paste(root.identifier, _id3(data, target, set(), nclasses))
    # A side-effect of using the treelib library is we have a plethora of redundant root nodes to remove
    nodes = list(
        tree.filter_nodes(
            lambda x: not (isinstance(x.data, Branch) or isinstance(x.data, Leaf))
        )
    )
    for node in nodes:
        if node.identifier != root.identifier:
            tree.link_past_node(node.identifier)
    return tree
