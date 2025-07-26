import redner
import numpy as np

def test_sample_discrete_n():

    np.random.seed(0)
    np_probs = np.random.rand(5)
    sum = np.sum(np_probs)
    np_probs /= sum
    probs = redner.double_ptr(0)
    probs.allocate(5)
    for i in range(5):
        probs.set_index(i, np_probs[i])

    n_samples = 1000000
    counts = np.zeros(5)
    
    for i in range(n_samples):
        sample = np.random.rand(1)[0]
        res = redner.sample_discrete_n_py(probs, 5, sample)
        counts[res] += 1 / n_samples

    for i in range(5):
        assert np.allclose(counts, np_probs, atol=0.01)