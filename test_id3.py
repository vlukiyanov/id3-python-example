import numpy as np
from pytest import approx

from id3 import entropy, gain

# column 0 is Outlook: 0 -> Sunny, 1 -> Overcast, 2 -> Rain
# column 1 is Temperature: 0 -> Cool, 1 -> Mild, 2 -> Hot
# column 2 is Humidity: 0 -> Normal, 1 -> High
# column 3 is Wind: 0 -> Weak, 1 -> Strong
# column 4 is PlayTennis: 0 -> No, 1 -> Yes

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


def test_entropy():
    assert entropy([1, 0, 0]) > 0
    assert entropy([1, 0, 0]) < 1
    assert entropy([1, 1, 1]) == approx(0.0)
    assert entropy([1, 0, 1, 0]) == approx(1.0)


def test_gain():
    data = test_playtennis[:, 0:4]
    target = test_playtennis[:, 4]
    assert gain(data, target, 0) == approx(0.246, abs=1e-3)
    assert gain(data, target, 1) == approx(0.029, abs=1e-3)
    assert gain(data, target, 2) == approx(0.151, abs=1e-3)
    assert gain(data, target, 3) == approx(0.048, abs=1e-3)


def test_id3_helper():
    pass
