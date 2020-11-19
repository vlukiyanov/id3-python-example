import numpy as np

from id3 import id3

test_playtennis = np.array(
    [
        [0, 2, 1, 0, 0],
        [0, 2, 1, 1, 0],
        [1, 2, 1, 0, 1],
        [2, 1, 1, 0, 1],
        [2, 0, 0, 0, 1],
        [2, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [2, 1, 0, 0, 1],
        [0, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 2, 0, 0, 1],
        [2, 1, 1, 1, 0],
    ]
)

test_nclasses = [3, 3, 2, 2]

data = test_playtennis[:, 0:4]
target = test_playtennis[:, 4]
tree = id3(data, target, test_nclasses)
tree.show()
