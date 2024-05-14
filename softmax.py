import numpy as np
import math

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp_l = [math.exp(x) for x in L]
    deno = sum(exp_l)
    probs = [x/deno for x in exp_l]
    return probs