import numpy as np


def msg_to_seq(msg, Tr):
    seq = []
    for b in msg:
        seq += [b] + [0] * Tr if b > 0 else [b]
    return seq


def seq_to_msg(seq, Tr):
    length = len(seq)
    msg = []
    i = 0
    while i < length:
        b = seq[i]
        msg.append(b)
        if b > 0:
            i += Tr + 1
        else:
            i += 1
    return msg

