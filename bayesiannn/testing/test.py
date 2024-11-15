class A:
    def __init__(self, x: "B"):
        pass


class B:
    def __init__(self, x: A):
        pass
