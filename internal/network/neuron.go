package network

import "mlp/pkg/vector"

type neuron struct {
	id       uint
	weights  vector.Vector[float64]
	bias     float64
	activate activationFunc
}

func newNeuron(id uint, activationFunc activationFunc) *neuron {
	return &neuron{
		id:       id,
		activate: activationFunc,
	}
}

func (n *neuron) forward(input vector.Vector[float64]) float64 {

	if n.weights == nil {
		n.bias = 1
		n.weights = make(vector.Vector[float64], len(input))
		for i := range n.weights {
			n.weights[i] = 1
		}
	}

	dot := n.weights.Dot(input) + n.bias

	return activate(n.activate, dot)
}

func (n *neuron) applyLoss(deltaW vector.Vector[float64], deltaB float64) {
	n.weights = n.weights.Sub(deltaW)
	n.bias = n.bias - deltaB
}

func activate(activation activationFunc, input float64) float64 {
	switch activation {
	case SigmoidActivation:
		return sigmoidActivation(input)
	case SoftmaxActivation:
		return input
	}
	return 0
}

func derivative(activation activationFunc, input float64) float64 {
	switch activation {
	case SigmoidActivation:
		return sigmoidDerivative(input)
	case SoftmaxActivation:
		return input
	}
	return 0
}
