from collections import OrderedDict


def time_range(start_time, end_time, interval):
    """Range of times specified by start_time and end_time (datetime)
    and interval (timedelta).
    """
    t = start_time
    while t < end_time:
        yield t
        t += interval


class CacheDict(OrderedDict):
    """Dict with a limited size, ejecting oldest items as needed."""

    def __init__(self, *args, cache_size=None, **kwargs):
        assert cache_size > 0
        self.cache_size = cache_size

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        if self.cache_size is not None:
            while len(self) > self.cache_size:
                old_key = next(iter(self))
                super().__delitem__(old_key)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val
