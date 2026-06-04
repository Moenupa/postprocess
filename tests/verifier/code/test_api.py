import requests


def test_health():
    response = requests.get("http://localhost:1234/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_program_simple_success():
    response = requests.post(
        "http://localhost:1234/test_program",
        json={
            "program": "def add(a, b): return a + b",
            "tests": [
                "assert add(1, 2) == 3",
                "assert add(-1, 1) == 0",
                "assert add(0, 0) == 1",
            ],
            "max_execution_time": 1.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == [1, 1, 0]
    assert "runtimes" in data


def test_program_simple_failed():
    response = requests.post(
        "http://localhost:1234/test_program",
        json={
            "program": "def add(a, b): return a + b + c",
            "tests": [
                "assert add(1, 2) == 3",
                "assert add(-1, 1) == 0",
                "assert add(0, 0) == 1",
            ],
            "max_execution_time": 1.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == [0] * 3
    assert data["runtimes"] == [-1] * 3
    for err in data["errors"]:
        assert "NameError: name 'c' is not defined" in err


INF_LOOP_PROGRAM = """
def add(a, b):
    while True:
        pass
    return a + b
"""


def test_program_simple_timeout():
    response = requests.post(
        "http://localhost:1234/test_program",
        json={
            "program": INF_LOOP_PROGRAM,
            "tests": ["assert add(1, 2) == 3"],
            "max_execution_time": 1.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == [0]
    assert data["runtimes"] == [-1]
    for err in data["errors"]:
        assert "Time Limit Exceeded" in err
