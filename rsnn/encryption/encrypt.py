import numpy as np

from rsnn.encryption.utils import msg_to_seq
from ..sampling.bffs import backward_filtering_forward_sampling


def encryption(msg, key_len, num_channels, complexity, Tr):
    c_seqs, c_seqs_len = split_msg(msg, Tr, num_channels)

    keys = []
    for c in range(num_channels):
        keys.append(generate_key(c_seqs[c], key_len, Tr))

    # learn sequence ...
    rsnn = ...

    return rsnn, keys, c_seqs_len


def split_msg(msg, Tr, num_channels):
    c_msgs = np.array_split(msg, num_channels)

    c_seqs_len = []

    for c in range(num_channels):
        seq = msg_to_seq(c_msgs[c], Tr)
        c_seqs_len.append(len(seq))

    c_seqs = []
    for c in range(num_channels):
        c_seqs.append(msg_to_seq(c_msgs[c], Tr) + [0] * max(c_seqs_len) - c_seqs_len[c])

    return c_seqs, c_seqs_len


def generate_key(seq, key_len, Tr):
    mapping = lambda seq_: 0 if np.sum(seq) < 1 else np.argmax(seq_) + 1

    s_left = mapping(seq[-Tr:])
    s_right = mapping(seq[0:Tr])

    return backward_filtering_forward_sampling(
        key_len + 2 * Tr, Tr, s_lim=(s_left, s_right), loop=False
    )[Tr : Tr + key_len]

