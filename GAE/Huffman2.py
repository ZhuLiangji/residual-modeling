import torch
import heapq
from collections import defaultdict, namedtuple
from itertools import count

class Node:
    def __init__(self, symbol, freq, left, right):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

def build_huffman_tree(unique, counts):
    """Build Huffman tree using symbol frequencies."""
    heap = []
    counter = count()  # ensures uniqueness

    for u, c in zip(unique.tolist(), counts.tolist()):
        heapq.heappush(heap, (c, next(counter), Node(u, c, None, None)))

    while len(heap) > 1:
        c1, _count1, left = heapq.heappop(heap)
        c2, _count2, right = heapq.heappop(heap)
        new_node = Node(None, c1 + c2, left, right)
        heapq.heappush(heap, (c1 + c2, next(counter), new_node))

    return heapq.heappop(heap)[2]

def build_codebook(node, prefix="", codebook=None):
    """Traverse the tree to assign binary codes."""
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        build_codebook(node.left, prefix + "0", codebook)
        build_codebook(node.right, prefix + "1", codebook)
    return codebook

def encoding_unsign_integer(data):
    """Encode data using the codebook."""
    unique, counts = torch.unique(data, return_counts=True)
    unique, counts = unique.cpu().numpy(), counts.cpu().numpy()

    codebook = build_codebook(build_huffman_tree(unique, counts))

    # Compute total bits needed
    bit_count = sum(len(codebook[symbol]) * count for symbol, count in zip(unique, counts)) + len(unique)* 16*2

    return bit_count


def huffman_decode(encoded, codebook):
    """Decode a bit string using the codebook."""
    # Reverse codebook
    decodebook = {v: k for k, v in codebook.items()}
    curr = ""
    decoded = []
    for bit in encoded:
        curr += bit
        if curr in decodebook:
            decoded.append(decodebook[curr])
            curr = ""
    return torch.tensor(decoded)

# # Example Usage
# if __name__ == "__main__":
#     data = torch.randint(0, 8, (1000,))
#     tree = build_huffman_tree(data)
#     codebook = build_codebook(tree)

#     encoded = huffman_encode(data, codebook)
#     decoded = huffman_decode(encoded, codebook)

#     print(f"Original: {data.tolist()[:20]}")
#     print(f"Decoded : {decoded.tolist()[:20]}")
#     print(f"Compression Ratio: {len(encoded) / (len(data) * 8):.4f}")
