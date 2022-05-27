import math
import torch
import numpy as np

"""
This script takes a number of tree nets and determines a comparable
single network in terms of parameter counts.
"""

baseline_chans = [
    [8,16,32,48],
    [16,32,64,96],
    [32,64,128,192],
    [32,64,128,256,512],
]

hyps = {
    "n_cnns": 128,
    "inpt_shape": (3,32,32),
    "chans": baseline_chans[0],
    "ksizes": 2,
    "strides": 1,
    "paddings": 0,
    "lnorm": True,
    "h_size": 256,
    #"agg_fxn": AvgOverDim,
    "agg_dim": 128,
    "share_base": False,
    "seq_len": 4,
    "cls": True,
    "output_type": "gapooling",
}

def count_params(chans, h_size, ksize=2):
    count = 0
    for i in range(len(chans)-1):
        count += math.prod([ chans[i], chans[i+1], ksize**2 ])
    count += math.prod([chans[-1], h_size])
    return count

h_size = hyps["h_size"]
leaf_count = count_params(hyps["chans"], hyps["h_size"])
print("Leaf Count:", leaf_count)

n_cnns = []
for chans in baseline_chans:
    count = count_params(chans, h_size)
    n = int(count//leaf_count)
    n_cnns.append(n)
print("N_CNN Counts:", n_cnns)


