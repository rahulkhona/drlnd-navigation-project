import pytest
from replay_buffer import ReplayBuffer

class TestReplayBuffer:
    def test_constructor(self):
        rb = ReplayBuffer()
        assert rb is not None
        assert rb.buffer_size == 1000
        assert rb.batch_size == 64
        assert len(rb) == 0

        rb = ReplayBuffer(seed=1)
        assert rb is not None
        assert rb.buffer_size == 1000
        assert rb.batch_size == 64
        assert len(rb) == 0

        with pytest.raises(ValueError):
            rb = ReplayBuffer(100, 1000)
    
    def test_add(self):
        rb = ReplayBuffer()
        rb.add([1], 0, 0.0, [2], False)
        assert len(rb) == 1

        for i in range(2000):
            rb.add([1], 0, 0.0, [2], False)

        assert len(rb) == rb.buffer_size

    def test_sample(self):
        rb = ReplayBuffer()
        rb.add([1], 0, 0.0, [2], False)
        assert len(rb) == 1

        for i in range(2000):
            rb.add([1], 0, 0.0, [2], False)

        assert len(rb) == rb.buffer_size
        samples = rb.sample()
        assert len(samples) == 5
        for i in range(len(samples)):
            assert len(samples[i]) == rb.batch_size

        samples = rb.sample(2)
        assert len(samples) == 5
        for i in range(len(samples)):
            assert len(samples[i]) == 2