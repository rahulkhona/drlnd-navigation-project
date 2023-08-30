import pytest
from priority_replay_buffer import PriorityReplayBuffer

class TestPriorityReplayBuffer:
    def test_constructor(self):
        rb = PriorityReplayBuffer()
        assert rb is not None
        assert rb.buffer_size == 1000
        assert rb.batch_size == 64
        assert rb.uniformity == 0.01
        assert len(rb) == 0

        rb = PriorityReplayBuffer(seed=1)
        assert rb is not None
        assert rb.buffer_size == 1000
        assert rb.batch_size == 64
        assert rb.uniformity == 0.01
        assert len(rb) == 0

        with pytest.raises(ValueError):
            rb = PriorityReplayBuffer(100, 1000)

        with pytest.raises(ValueError):
            rb = PriorityReplayBuffer(uniformity=2)
    
    def test_add(self):
        rb = PriorityReplayBuffer()
        rb.add([1], 0, 0.0, [2], False, 10)
        assert len(rb) == 1

        for i in range(2000):
            rb.add([1], 0, 0.0, [2], False, 10)

        assert len(rb) == rb.buffer_size

    def test_sample(self):
        rb = PriorityReplayBuffer(buffer_size=5, batch_size=3)
        rb.add([1], 0, 0.0, [2], False, 30)
        rb.add([2], 0, 0.0, [2], False, 20)
        rb.add([3], 0, 0.0, [2], False, 10)
        rb.add([4], 0, 0.0, [2], False, 6)
        rb.add([5], 0, 0.0, [2], False, 4)

        samples = rb.sample(beta=0.3)
        assert len(samples) == 6
        for i in range(len(samples)):
            assert len(samples[i]) == rb.batch_size

        rb = PriorityReplayBuffer(buffer_size=5, batch_size=2)
        rb.add([1], 0, 0.0, [2], False, 30)
        rb.add([2], 0, 0.0, [2], False, 20)
        rb.add([3], 0, 0.0, [2], False, 10)
        rb.add([4], 0, 0.0, [2], False, 6)
        rb.add([5], 0, 0.0, [2], False, 4)
        samples = rb.sample(beta=0.3)
        assert len(samples) == 6
        for i in range(len(samples)):
            assert len(samples[i]) == 2