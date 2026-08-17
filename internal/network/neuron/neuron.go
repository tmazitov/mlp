package neuron

import (
	"mlp/pkg/vector"
)

type NeuronParams struct {
	NIn          uint
	NOut         uint
	InitType     WeightInitFunc
	LearningRate float64
}

type Neuron struct {
	id      uint
	bias    float64
	weights vector.Vector[float64]
	params  NeuronParams
}

func NewNeuron(id uint, params NeuronParams) *Neuron {

	n := &Neuron{
		id:     id,
		params: params,
	}

	n.initWeights()

	return n
}

func (n *Neuron) initWeights() {

	switch n.params.InitType {
	case XavierInitFunc:
		n.bias = xavierInit(n.params.NIn, n.params.NOut)
		n.weights = xavierInitVector(n.params.NIn, n.params.NOut)
	case HeInitFunc:
		n.bias = heInit(n.params.NIn)
		n.weights = heInitVector(n.params.NIn)
	}
}

func (n *Neuron) ApplyLoss(deltaW vector.Vector[float64], deltaB float64) {
	n.weights = n.weights.Sub(deltaW.Scl(n.params.LearningRate))
	n.bias = n.bias - deltaB*n.params.LearningRate
}

func (n Neuron) Weight() vector.Vector[float64] { return n.weights }

func (n Neuron) Sum(input vector.Vector[float64]) float64 {
	return n.weights.Dot(input) + n.bias
}
