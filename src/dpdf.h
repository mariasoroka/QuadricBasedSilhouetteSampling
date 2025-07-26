#pragma once

#include "buffer.h"

// Sample out of a discrete probability distribution. Reuse the sample.
inline int sample_discrete_n(const Real *probs, int n, Real &sample){
    Real prob_sum = 0;
    for(int i = 0; i < n; i++) {
        prob_sum += probs[i];
        if (sample < prob_sum) {
            Real l_bound = prob_sum - probs[i];
            sample = (sample - l_bound) / probs[i];
            return i;
        }
    }
    return 0;
}

// Sample out of a discrete probability distribution with 4 elements. Reuse the sample.
inline int sample_discrete_4(const Real *probs, Real &sample) {
    Real prob_0 = probs[0] + probs[1];
    Real prob_1 = probs[2] + probs[3];

    if (sample < prob_0) {
        sample /= prob_0;
        Real prob_00 = probs[0] / (probs[0] + probs[1]);
        Real prob_01 = probs[1] / (probs[0] + probs[1]);
        if (sample < prob_00) {
            sample /= prob_00;
            return 0;
        }
        else {
            sample = (sample - prob_00) / (1 - prob_00);
            return 1;
        }
    }
    else {
        sample = (sample - prob_0) / (1 - prob_0);
        Real prob_10 = probs[2] / (probs[2] + probs[3]);
        Real prob_11 = probs[3] / (probs[2] + probs[3]);
        if (sample < prob_10) {
            sample /= prob_10;
            return 2;
        }
        else {
            sample = (sample - prob_10) / (1 - prob_10);
            return 3;
        }
    }
}

inline int sample_discrete_n_py(ptr<double> probs_ptr, int n, Real &sample){
    Real *probs = probs_ptr.get_pointer();
    return sample_discrete_n(probs, n, sample);
}