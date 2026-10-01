package activation

import (
	"math"
	"mlp/pkg/vector"
)

type SoftMaxFunc struct{}

func (a SoftMaxFunc) Name() string { return "softmax" }

// Activate turns the layer's raw sums into a probability distribution.
//
// The largest sum is subtracted from every value first. That cancels out of
// the ratio — exp(z-m)/Σexp(zⱼ-m) is algebraically identical to
// exp(z)/Σexp(zⱼ) — but it keeps every exponent at or below zero, so a large
// sum can no longer overflow math.Exp to +Inf and turn the result into NaN.
func (a SoftMaxFunc) Activate(input vector.Vector[float64]) vector.Vector[float64] {

	if len(input) == 0 {
		return vector.Vector[float64]{}
	}

	max := input[0]
	for _, value := range input {
		if value > max {
			max = value
		}
	}

	var expSum float64
	exps := make(vector.Vector[float64], len(input))
	for i, value := range input {
		exps[i] = math.Exp(value - max)
		expSum += exps[i]
	}

	probability := make(vector.Vector[float64], len(input))
	for i, exp := range exps {
		probability[i] = exp / expSum
	}

	return probability
}

func (a SoftMaxFunc) Derivative(input vector.Vector[float64]) vector.Vector[float64] {
	return vector.Vector[float64]{}
}
