from queue import PriorityQueue
from random import random

token_probs = {"a": 0.5, "b": 0.25, "c": 0.125, "d": 0.125}


class TreeNode:
    def __init__(self, label, left=None, right=None):
        self.label = label
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

    def __str__(self):
        if self.left is None and self.right is None:
            return self.label
        return f"({self.label} {self.left} {self.right})"


def huffman_training(token_probs):
    treebank = PriorityQueue()
    for token in token_probs:
        treebank.put((token_probs[token], random(), TreeNode(token)))

    while not treebank.empty():
        p1, _, node1 = treebank.get()
        if treebank.empty():
            return node1
        p2, _, node2 = treebank.get()
        combined_prob = p1 + p2
        combo_node = TreeNode(node1.label + node2.label, left=node1, right=node2)
        treebank.put((combined_prob, random(), combo_node))


def extract_encodings(tree, prefix=""):
    if tree.is_leaf():
        return {tree.label: prefix}
    if tree.left is not None:
        left_encodings = extract_encodings(tree.left, prefix + "0")
    if tree.right is not None:
        right_encodings = extract_encodings(tree.right, prefix + "1")
    return left_encodings | right_encodings


def huffman(token_probs):
    tree = huffman_training(token_probs)
    return extract_encodings(tree)


# tree = huffman_training(token_probs)
# encodings = extract_encodings(tree)
# print(encodings)
