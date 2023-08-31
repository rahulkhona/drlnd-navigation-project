import pytest
from replay_buffer import PriorityReplayBuffer
import numpy as np
from hyper_parameter_providers import LinearChangeParameterProvider

class TestPriorityReplayBuffer:
    def test_constructor(self):
        rb = PriorityReplayBuffer(LinearChangeParameterProvider(0.1, 1, 0.1))
        assert rb is not None
        assert rb.buffer_size == 1000
        assert rb.batch_size == 64
        assert rb.uniformity == 0.01
        assert len(rb) == 0

        rb = PriorityReplayBuffer(LinearChangeParameterProvider(0.1, 1, 0.1),seed=1)
        assert rb is not None
        assert rb.buffer_size == 1000
        assert rb.batch_size == 64
        assert rb.uniformity == 0.01
        assert len(rb) == 0

        with pytest.raises(ValueError):
            rb = PriorityReplayBuffer(LinearChangeParameterProvider(0.1, 1, 0.1), buffer_size=100, batch_size=1000)

        with pytest.raises(ValueError):
            rb = PriorityReplayBuffer(LinearChangeParameterProvider(0.1, 1, 0.1), uniformity=2)
    
    def test_add(self):
        rb = PriorityReplayBuffer(LinearChangeParameterProvider(0.1, 1, 0.1))
        rb.add(np.array([1]), 0, 0.0, np.array([2]), False)
        assert len(rb) == 1

        for i in range(2000):
            rb.add(np.array([1]), 0, 0.0, np.array([2]), False)

        assert len(rb) == rb.buffer_size

    def test_sample(self):
        rb = PriorityReplayBuffer(LinearChangeParameterProvider(0.1, 1, 0.1), buffer_size=5, batch_size=3)
        rb.add(np.array([1]), 0, 0.0, np.array([2]), False)
        rb.add(np.array([2]), 0, 0.0, np.array([2]), False)
        rb.add(np.array([3]), 0, 0.0, np.array([2]), False)
        rb.add(np.array([4]), 0, 0.0, np.array([2]), False)
        rb.add(np.array([5]), 0, 0.0, np.array([2]), False)

        samples, importances, indices = rb.sample()
        assert len(samples) == 5
        for i in range(len(samples)):
            assert len(samples[i]) == rb.batch_size

        print(type(importances), type(samples[0]), type(indices))
        assert len(samples[0]) == len(importances)
        assert len(samples[0]) == len(indices)

    def test_update(self):
        rb = PriorityReplayBuffer(LinearChangeParameterProvider(0.1, 1, 0.1), buffer_size=5, batch_size=3)
        rb.add(np.array([1]), 0, 0.0, np.array([2]), False)
        rb.add(np.array([2]), 0, 0.0, np.array([2]), False)
        rb.add(np.array([3]), 0, 0.0, np.array([2]), False)
        rb.add(np.array([4]), 0, 0.0, np.array([2]), False)
        rb.add(np.array([5]), 0, 0.0, np.array([2]), False)

        samples, importances, indices = rb.sample()
        print(importances)
        importances = importances + np.random.normal(0, 1, len(importances))
        print(importances)
        rb.update_errors(importances, indices)

