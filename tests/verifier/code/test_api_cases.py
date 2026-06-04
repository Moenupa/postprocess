import requests

KEY_CODING = {
    "data_source": "code_rlvr_mixture_dpo",
    "prompt": "Write a function `keys(obj)` that takes an object `obj` as input and returns a list of the object's own keys, excluding any inherited keys from its prototype chain. The keys should be returned in the order they are defined in the object. For example, if `obj` is an instance of a class with its own attributes, the function should return only those attribute names. Do not include any keys that start with '__'.",
    "ability": "CODE",
    "reward_model": {
        "style": "code",
        "ground_truth": "[\"class D: pass\\nassert keys(D()) == []\", \"class E:\\n    def __init__(self):\\n        self.x = 10\\n        self.y = 20\\nassert keys(E()) == ['x', 'y']\", \"class G:\\n    def __init__(self):\\n        self.a = 'alpha'\\n        self.b = 'beta'\\nassert keys(G()) == ['a', 'b']\", \"class J:\\n    def __init__(self):\\n        self.one = 1\\n        self.two = 2\\n        self.three = 3\\nassert keys(J()) == ['one', 'two', 'three']\", \"class K:\\n    def __init__(self):\\n        self.name = 'test'\\n        self.value = 42\\nassert keys(K()) == ['name', 'value']\", \"class L:\\n    def __init__(self):\\n        self.a = 'A'\\n        self.b = 'B'\\n        self.c = 'C'\\nassert keys(L()) == ['a', 'b', 'c']\", \"class M:\\n    def __init__(self):\\n        self.attr1 = 'foo'\\n        self.attr2 = 'bar'\\n        self.attr3 = None\\nassert keys(M()) == ['attr1', 'attr2', 'attr3']\", \"class O:\\n    def __init__(self):\\n        self.flag = True\\n        self.count = 0\\nassert keys(O()) == ['flag', 'count']\", \"class P:\\n    def __init__(self):\\n        self.first = 'first'\\n        self.second = 'second'\\nassert keys(P()) == ['first', 'second']\", \"class R:\\n    def __init__(self):\\n        self.alpha = 'A'\\n        self.beta = 'B'\\n        self.gamma = 'C'\\nassert keys(R()) == ['alpha', 'beta', 'gamma']\", \"class S:\\n    def __init__(self):\\n        self.num1 = 1\\n        self.num2 = 2\\nassert keys(S()) == ['num1', 'num2']\"]",
    },
    "extra_info": {
        "split": "train",
        "index": "rlvr_acecoder_filtered_filtered/request-603-68",
    },
}
KEY_CODING_SOLUTION = """
def keys(obj):
    return [k for k in obj.__dict__.keys() if not k.startswith('__')]
"""


def test_api_program_keys():
    response = requests.post(
        "http://localhost:1234/test_program",
        json={
            "program": KEY_CODING_SOLUTION,
            "tests": [
                "class D: pass\nassert keys(D()) == []",
                "class E:\n    def __init__(self):\n        self.x = 10\n        self.y = 20\nassert keys(E()) == ['x', 'y']",
                "class G:\n    def __init__(self):\n        self.a = 'alpha'\n        self.b = 'beta'\nassert keys(G()) == ['a', 'b']",
                "class J:\n    def __init__(self):\n        self.one = 1\n        self.two = 2\n        self.three = 3\nassert keys(J()) == ['one', 'two', 'three']",
                "class K:\n    def __init__(self):\n        self.name = 'test'\n        self.value = 42\nassert keys(K()) == ['name', 'value']",
                "class L:\n    def __init__(self):\n        self.a = 'A'\n        self.b = 'B'\n        self.c = 'C'\nassert keys(L()) == ['a', 'b', 'c']",
                "class M:\n    def __init__(self):\n        self.attr1 = 'foo'\n        self.attr2 = 'bar'\n        self.attr3 = None\nassert keys(M()) == ['attr1', 'attr2', 'attr3']",
                "class O:\n    def __init__(self):\n        self.flag = True\n        self.count = 0\nassert keys(O()) == ['flag', 'count']",
                "class P:\n    def __init__(self):\n        self.first = 'first'\n        self.second = 'second'\nassert keys(P()) == ['first', 'second']",
                "class R:\n    def __init__(self):\n        self.alpha = 'A'\n        self.beta = 'B'\n        self.gamma = 'C'\nassert keys(R()) == ['alpha', 'beta', 'gamma']",
                "class S:\n    def __init__(self):\n        self.num1 = 1\n        self.num2 = 2\nassert keys(S()) == ['num1', 'num2']",
            ],
            "max_execution_time": 5.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == [1] * 11
    assert data["errors"] == [""] * 11
    for runtime in data["runtimes"]:
        assert 0 < runtime <= 5.0


TOOTHPICK_CODING = """
Solve the following coding problem using the programming language python: Introduction It's been more than 20 minutes since the negligent waiter has taken your order for the house special prime tofu steak with a side of chili fries. Out of boredom, you start fiddling around with the condiments tray. To be efficient, you want to be familiar with the choice of sauces and spices before your order is finally served. You also examine the toothpick holder and try to analyze its inner workings when - yikes - the holder's lid falls off and all 23 picks lay scattered on the table. Being a good and hygiene oriented citizen, you decide not to just put them back in the holder. Instead of letting all the good wood go to waste, you start playing around with the picks. In the first "round", you lay down one toothpick vertically. You've used a total of one toothpick. In the second "round", at each end of the first toothpick, you add a perpendicular toothpick at its center point. You added two additional toothpicks for a total of three toothpicks. In the next rounds, you continue to add perpendicular toothpicks to each free end of toothpicks already on the table. With your 23 toothpicks, you can complete a total of six rounds: You wonder if you'd be able to implement this sequence in your favorite programming language. Because your food still hasn't arrived, you decide to take out your laptop and start implementing... Challenge Implement a script that returns the amount of toothpicks needed to complete n amount of rounds of the toothpick sequence. ``` 0 <= n <= 5000 ``` Hint You can attempt this brute force or get some inspiration from the math department. Write your solution by modifying this code: ```python def toothpick(n): ``` Your solution should implemented in the function "toothpick". The inputs will be passed to it and it should return the correct solution. Now solve the problem and return the code.
"""
TOOTHPICK_CODING_SOLUTION = """
_MAX_N = 5000
_OFF = _MAX_N + 1
_W = 2 * _OFF + 1

_rows = None
_frontier = None
_totals = [0]
_done = 0
_total = 0


def _init():
    global _rows, _frontier, _totals, _done, _total

    if _rows is not None:
        return

    _rows = [0] * _W

    def occupy(x, y):
        _rows[y + _OFF] |= 1 << (x + _OFF)

    # Round 1: one vertical toothpick centered at (0, 0)
    occupy(0, 0)
    occupy(0, 1)
    occupy(0, -1)

    # Free ends; next toothpicks placed there will be horizontal.
    # orientation: 0 = horizontal, 1 = vertical
    _frontier = [(0, 1, 0), (0, -1, 0)]

    _totals = [0, 1]
    _done = 1
    _total = 1


def toothpick(n):
    global _frontier, _done, _total

    if n == 0:
        return 0

    _init()

    while _done < n:
        _done += 1
        _total += len(_frontier)

        candidates = {}
        rows = _rows
        off = _OFF
        width = _W

        for x, y, orient in _frontier:
            if orient == 0:  # place horizontal toothpick
                endpoints = ((x - 1, y), (x + 1, y))
                next_orient = 1
            else:            # place vertical toothpick
                endpoints = ((x, y - 1), (x, y + 1))
                next_orient = 0

            for ex, ey in endpoints:
                xi = ex + off
                yi = ey + off

                if (rows[yi] >> xi) & 1:
                    continue

                key = yi * width + xi

                if key in candidates:
                    candidates[key] = -1
                else:
                    candidates[key] = next_orient

        new_frontier = []

        for key, orient in candidates.items():
            yi, xi = divmod(key, width)
            rows[yi] |= 1 << xi

            if orient != -1:
                new_frontier.append((xi - off, yi - off, orient))

        _frontier = new_frontier
        _totals.append(_total)

    return _totals[n]
"""


def test_api_program_toothpick():
    response = requests.post(
        "http://localhost:1234/test_program",
        json={
            "program": TOOTHPICK_CODING_SOLUTION,
            "tests": [
                "assert toothpick(0) == 0",
                "assert toothpick(3) == 7",
                "assert toothpick(16) == 171",
                "assert toothpick(32) == 683",
                "assert toothpick(49) == 1215",
                "assert toothpick(89) == 3715",
                "assert toothpick(327) == 52239",
                "assert toothpick(363) == 60195",
                "assert toothpick(366) == 62063",
                "assert toothpick(512) == 174763",
                "assert toothpick(656) == 209095",
                "assert toothpick(1038) == 699451",
                "assert toothpick(1052) == 700379",
                "assert toothpick(1222) == 757295",
                "assert toothpick(1235) == 762019",
                "assert toothpick(1302) == 832559",
                "assert toothpick(1735) == 1398915",
                "assert toothpick(1757) == 1443119",
                "assert toothpick(1974) == 2038207",
                "assert toothpick(2048) == 2796203",
            ],
            "max_execution_time": 5.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == [1] * 20
    assert data["errors"] == [""] * 20
