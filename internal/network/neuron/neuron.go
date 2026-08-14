package neuron

import (
	"mlp/pkg/vector"
)

type Neuron struct {
	id      uint
	bias    float64
	weights vector.Vector[float64]
}

func NewNeuron(id uint) *Neuron {
	return &Neuron{
		id: id,
	}
}

func (n *Neuron) ApplyLoss(deltaW vector.Vector[float64], deltaB float64) {
	n.weights = n.weights.Sub(deltaW)
	n.bias = n.bias - deltaB
}

func (n Neuron) Weight() vector.Vector[float64] { return n.weights }

func (n Neuron) Sum(input vector.Vector[float64]) float64 {
	return n.weights.Dot(input) + n.bias
}
