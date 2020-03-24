import numpy as np


def wedge(x):
    assert x.size in [3, 6]
    x = x.reshape(-1)
    if x.size == 3:
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    elif x.size == 6:
        return np.block([[wedge(x[3:]), x[:3].reshape(-1, 1)], [np.array([0, 0, 0, 0])]])


def cwedge(x):
    assert x.size == 6
    x = x.reshape(-1)
    return np.block([[wedge(x[3:]), wedge(x[:3])], [np.zeros((3, 3)), wedge(x[3:])]])


def cdot(x):
    x = x.reshape(-1)
    assert x.size == 4 and x[-1] == 1
    return np.block([[np.eye(3), -wedge(x[:3])], [np.zeros((1, 6))]])


def pi(q):
    assert q.ndim == 2 and q.shape[0] == 4
    return q / q[2, :]


def dpidq(q):
    assert q.size == 4
    q = q.reshape(-1)
    return (1 / q[2]) * np.array([[1, 0, -q[0] / q[2], 0], [0, 1, -q[1] / q[2], 0], [0, 0, 0, 0],
                                  [0, 0, -q[3] / q[2], 1]])
