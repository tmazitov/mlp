package activation

import (
	"math"
	"mlp/pkg/vector"
)

type SoftMaxFunc struct{}

func (a SoftMaxFunc) Name() string { return "softmax" }

func (a SoftMaxFunc) Activate(input vector.Vector[float64]) vector.Vector[float64] {

	var expSum float64
	for _, value := range input {
		expSum += math.Exp(value)
	}

	var probability = make(vector.Vector[float64], 0, len(input))
	for _, value := range input {
		probability = append(probability, math.Exp(value)/expSum)
	}

	return probability
}

func (a SoftMaxFunc) Derivative(input vector.Vector[float64]) vector.Vector[float64] {
	return vector.Vector[float64]{}
}
