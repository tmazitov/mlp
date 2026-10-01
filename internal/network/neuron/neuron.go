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
	id        uint
	bias      float64
	weights   vector.Vector[float64]
	params    NeuronParams
	lossCache lossCache
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

func (n *Neuron) AddBatchLoss(deltaW vector.Vector[float64], deltaB float64) {
	n.lossCache.AddRecord(deltaW, deltaB)
}

func (n *Neuron) ApplyLoss() {

	avgDeltaWeights, avgDeltaBias := n.lossCache.GetAverage()
	defer n.lossCache.Clear()

	n.weights = n.weights.Sub(avgDeltaWeights.Scl(n.params.LearningRate))
	n.bias = n.bias - avgDeltaBias*n.params.LearningRate

}

func (n Neuron) Weight() vector.Vector[float64] { return n.weights }
func (n Neuron) Bias() float64                  { return n.bias }

func (n Neuron) Sum(input vector.Vector[float64]) float64 {
	return n.weights.Dot(input) + n.bias
}

// FromWeights rebuilds a neuron with known weights and bias, skipping the
// random initialisation NewNeuron performs. Used when loading a trained
// model: the saved numbers are the whole point, and re-rolling them would
// throw the training away.
func FromWeights(id uint, weights vector.Vector[float64], bias float64) *Neuron {
	return &Neuron{
		id:      id,
		weights: weights,
		bias:    bias,
	}
}
