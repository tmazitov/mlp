package neuron

import (
	"mlp/pkg/vector"
)

type NeuronParams struct {
	NIn          uint
	NOut         uint
	InitType     WeightInitFunc
	LearningRate float64
	Optimizer    OptimizerFunc
}

type Neuron struct {
	id        uint
	bias      float64
	weights   vector.Vector[float64]
	params    NeuronParams
	lossCache lossCache

	// optimizer is per-neuron: momentum and squared-gradient averages are
	// state about this neuron's own parameters.
	optimizer optimizer
}

func NewNeuron(id uint, params NeuronParams) *Neuron {

	n := &Neuron{
		id:        id,
		params:    params,
		optimizer: newOptimizer(params.Optimizer, params.LearningRate),
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

	stepWeights, stepBias := n.optimizer.step(avgDeltaWeights, avgDeltaBias)

	n.weights = n.weights.Sub(stepWeights)
	n.bias = n.bias - stepBias
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
		id:        id,
		weights:   weights,
		bias:      bias,
		optimizer: newOptimizer(SGDOptimizer, 0),
	}
}

// SetParameters overwrites the weights and bias in place, used to restore
// a snapshot. The optimizer's own state is deliberately left alone: a
// restore happens at the end of a run, when nothing will step again.
func (n *Neuron) SetParameters(weights vector.Vector[float64], bias float64) {
	n.weights = weights
	n.bias = bias
}
