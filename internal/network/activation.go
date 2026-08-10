package network

import (
	"math"
	"mlp/pkg/vector"
)

type activationFunc string

var (
	SoftmaxActivation activationFunc = "softmax"
	SigmoidActivation activationFunc = "sigmoid"
)

func sigmoidActivation(input float64) float64 {
	return 1 / (1 + math.Exp(-input))
}

func sigmoidDerivative(output float64) float64 {
	return output * (1 - output)
}

func softmaxActivation(input vector.Vector[float64]) vector.Vector[float64] {

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
