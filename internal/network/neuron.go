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
		weights:  make(vector.Vector[float64], 0),
		bias:     1,
		activate: activationFunc,
	}
}

func (n neuron) forward(input vector.Vector[float64]) float64 {

	dot := n.weights.Dot(input) + n.bias
	switch n.activate {
	case SigmoidActivation:
		return sigmoidActivation(dot)
	case SoftmaxActivation:
		return dot
	}
	return 0
}
