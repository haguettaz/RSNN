import numpy as np
from rsnn.encryption.utils import seq_to_msg


def decrypt(rsnn, keys, c_seqs_len, Tr):
    c_seq = ...  # init with keys
    for _ in range(max(c_seqs_len)):
        c_seq = ...  # update until we get the whole message

    msg = []
    for c in range(rsnn.num_neurons):
        msg += seq_to_msg(c_seq[c][: c_seqs_len[c]], Tr)

    return msg
