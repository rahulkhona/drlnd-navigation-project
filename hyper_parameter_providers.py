import numpy as np
import typing

class HyperParameterProvider:
    def __init__(self):
        self.value = None
        self.increment = 0
    def get(self)->float|int:
        raise NotImplemented("This is abstract class, use a concrete subclass")

class ConstantParameterProvider:
    def __init__(self, value:int|float):
        super(ConstantParameterProvider, self).__init__()
        self.value = value
    def get(self)->float|int:
        self.increment += 1
        return self.value
    
class LinearAnealingParameterProvider(HyperParameterProvider):
    def __init__(self, min:float=0, max:float=0):
        super(LinearAnealingParameterProvider, self).__init__(max)
        if min > max:
            raise ValueError("Min should be smaller than max")
        self.min = min
        self.max = max
    def get(self)->float:
        self.increment += 1
        return max(self.max / self.increment, self.min)

class ExponentialDecayParameterProvider(LinearAnealingParameterProvider):
    def __init__(self, min:float, max:float, decay_rate:float):
        super(ExponentialDecayParameterProvider, self).__init__(self, min, max)
        if decay_rate < 0 or decay_rate > 1:
            raise ValueError("Decay Rate should be between 0 and 1")
        self.decay_rate = decay_rate

    def get(self)->float:
        self.increment += 1
        return max(self.max * np.exp(-self.decay_rate * self.increment), self.min)
        #return max(max * (1 - self.decay_rate)**self.increment, min)

class ExponentialGrowthParameterProvider(LinearAnealingParameterProvider):
    def __init__(self, min:float, max:float, growth_rate:float):
        super(ExponentialGrowthParameterProvider, self).__init__(self, min, max)
        if growth_rate < 0 or growth_rate > 1:
            raise ValueError("growth Rate should be between 0 and 1")
        self.growth_rate = growth_rate
    
    def get(self)->float:
        self.increment += 1
        return min(self.max * np.exp(self.growth_rate * self.increment), self.min)
        #return min((self.max + self.growth_rate))**self.increment, self.max)

class LinearGrowthParameterProvider(LinearAnealingParameterProvider):
    def __init__(self, min:float, max:float, slope:float):
        super(LinearGrowthParameterProvider, self).__init__(self, min, max)
        self.slope = slope

    def get(self)->float:
        self.increment += 1
        return min(self.max, self.min + self.increment * self.slope)
