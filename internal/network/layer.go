package network

import "mlp/pkg/vector"

type Layer struct {
	neurons    []*neuron
	activation activationFunc
}

func NewLayer(neuronCount uint, activation activationFunc) *Layer {

	neurons := make([]*neuron, 0, neuronCount)
	for i := range neuronCount {
		neurons = append(neurons, newNeuron(i, activation))
	}

	return &Layer{
		neurons:    neurons,
		activation: activation,
	}
}

func (l Layer) forwardValues(inputs vector.Vector[float64]) vector.Vector[float64] {

	// Do some stuff here with values
	activations := make(vector.Vector[float64], 0, len(l.neurons))

	// Apply input vector on each neuron
	for _, neuron := range l.neurons {
		activations = append(activations, neuron.forward(inputs))
	}

	if l.activation == SoftmaxActivation {
		return softmaxActivation(activations)
	}

	return activations
}
