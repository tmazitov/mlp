package activation

import (
	"math"
	"mlp/pkg/vector"
)

type SigmoidFunc struct{}

func (f SigmoidFunc) Name() string { return "sigmoid" }

func (f SigmoidFunc) Activate(input vector.Vector[float64]) vector.Vector[float64] {

	result := make(vector.Vector[float64], len(input))

	for i, inputElem := range input {
		result[i] = 1 / (1 + math.Exp(-inputElem))
	}

	return result
}

func (f SigmoidFunc) Derivative(input vector.Vector[float64]) vector.Vector[float64] {

	result := make(vector.Vector[float64], len(input))

	for i, inputElem := range input {
		result[i] = inputElem * (1 - inputElem)
	}

	return result
}
