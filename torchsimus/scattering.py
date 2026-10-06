import torch

def single_scatter(G, reflectivity, A):
    tx = G @ A
    scattered = reflectivity[:, None] * tx
    rx = G.transpose(-1, -2) @ scattered
    return rx

def scatter_operator(G, reflectivity, A):
    tx = G @ A
    scattered = reflectivity[:, None] * tx
    return G.transpose(-1, -2) @ scattered
