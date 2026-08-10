package network

import (
	"math"
	"mlp/pkg/vector"
)

type Layer struct {
	neurons    []*neuron
	activation activationFunc
	cache      forwardCache
}

func NewLayer(neuronCount uint, activation activationFunc) *Layer {

	neurons := make([]*neuron, 0, neuronCount)
	for i := range neuronCount {
		neurons = append(neurons, newNeuron(i, activation))
	}

	return &Layer{
		neurons:    neurons,
		activation: activation,
		cache:      newForwardCache(int(neuronCount)),
	}
}

// forwardValues allows to pass forward input vector and, using it, calculate
// its activation vector (a) and output vector (z). Store calculated values
// in cache structure for farther needs (local loss calculations).
//
// - input, hidden layers -> respond with output vector
//
// - output layer		  -> respond with activation vector (for farther percent calculation and evaluation)
func (l *Layer) forwardValues(inputs vector.Vector[float64]) vector.Vector[float64] {

	// Apply input vector on each neuron
	// and record as an element of activation vector
	for i, neuron := range l.neurons {
		l.cache.activation[i] = neuron.forward(inputs)
	}

	// If layer type is not outer
	if l.activation == SoftmaxActivation {
		l.cache.output = softmaxActivation(l.cache.activation)
		return l.cache.output
	}

	l.cache.output = l.cache.activation
	return l.cache.activation
}

func (l Layer) cleanCache() {
	l.cache.clean()
}

func (l Layer) calcWeightLossSum(lossVector vector.Vector[float64], index int) float64 {
	var sum float64

	for i, neuron := range l.neurons {
		sum = math.FMA(neuron.weights[index], lossVector[i], sum)
	}
	return sum
}

func (l Layer) calcDerivative(index int) float64 {
	currentValue := l.cache.output[index]

	return derivative(l.activation, currentValue)
}

func (l Layer) applyLoss(lossVector, inputs vector.Vector[float64]) {
	for i, neuron := range l.neurons {

		deltaW := inputs.Scl(lossVector[i])
		deltaB := lossVector[i]

		neuron.applyLoss(deltaW, deltaB)
	}
}
