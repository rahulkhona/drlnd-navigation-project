import numpy as np
import typing

class HyperParameterProvider:
    def __init__(self):
        self.value = None
        self.increment = 0

    def next(self):
        self.increment += 1
        return self.increment

    def get(self)->float|int:
        raise NotImplemented("This is abstract class, use a concrete subclass")

class ConstantParameterProvider(HyperParameterProvider):
    def __init__(self, value:int|float):
        super(ConstantParameterProvider, self).__init__()
        self.value = value
    def get(self)->float|int:
        self.increment += 1
        return self.value
class LinearChangeParameterProvider(HyperParameterProvider):
    def __init__(self, start:float, limit:float, slope:float):
        super(LinearChangeParameterProvider, self).__init__()
        self.start = start
        self.limit = limit
        self.rate = slope
        if self.rate < 0:
            if start < limit:
                raise ValueError("For negative slopes, start must be larger than limi")
        elif self.rate > 0:
            if start > limit:
                raise ValueError("For positive slopes, start must be smaller than limi")
        elif self.rate == 0:
            if start != limit:
                raise ValueError("For 0 slope, start must be equal to limit")

    def compute_value(self):
        timestep = self.next()
        return self.start + self.rate * timestep

    def get(self)->float:
        if np.isclose(self.start, self.limit):
            return self.start
        value = self.compute_value()
        if self.rate < 0.0:
            return max(self.limit, value)
        else:
            return min(self.limit, value)

class ExponentialChangeParameterProvider(LinearChangeParameterProvider):
    def __init__(self, start:float, limit:float, rate:float):
        super(ExponentialChangeParameterProvider, self).__init__(start, limit, rate)
    
    def compute_value(self):
        timestep = self.next()
        return self.start * np.exp(self.rate * timestep)

class ScheduledChangeParameterProvider(HyperParameterProvider):
    def __init__(self, timesteps:[int], values:[float]):
        super(ScheduledChangeParameterProvider, self).__init__()
        if len(timesteps) != len(values):
            raise ValueError("number of timesteps should be same as number of values")
        self.q = []
        for t,v in zip(timesteps, values):
            self.q.append((t, v))

    def get(self)->float:
        timestep = self.next()
        i = 0
        while i < timestep and i < len(self.q):
            v = self.q[i][1]
            i+=1
        return v
