import time

class _TimingBase:

    timing = {}

    @classmethod
    def get(cls):
        return cls.timing

    @classmethod
    def reset(cls):
        cls.timing = {}

    def __init__(self, name):
        self._name = name

    def __enter__(self):
        self._t0 = time.time()

    def __exit__(self, *exc_args):
        self.__class__.timing[self._name] = time.time() - self._t0

registry = {}

def makeOrGetTimingClass(name: str):
    if name not in registry:
        registry[name] = type(f'Timing_{name}', (_TimingBase,), {'timing': {}})
    return registry[name]

class Global(_TimingBase):
    """
    A global timing class that can be used to track timing anywhere
    """
    timing = {}

