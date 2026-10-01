package network

import (
	"fmt"

	"mlp/internal/network/activation"
	"mlp/internal/network/neuron"
	"mlp/pkg/vector"
)

type activationFunc interface {
	Name() string
	Activate(vector.Vector[float64]) vector.Vector[float64]
	Derivative(vector.Vector[float64]) vector.Vector[float64]
}

type Layer struct {
	neurons    []*neuron.Neuron
	activation activationFunc
	cache      layerCache
}

func NewLayer(neuronCount uint, activation activationFunc, neuronParams neuron.NeuronParams) (*Layer, error) {

	neurons := make([]*neuron.Neuron, 0, neuronCount)
	for i := range neuronCount {
		neurons = append(neurons, neuron.NewNeuron(i, neuronParams))
	}

	return &Layer{
		neurons:    neurons,
		activation: activation,
		cache:      newLayerCache(int(neuronCount)),
	}, nil
}

// ActivationFunc Section

func (l Layer) activate(input vector.Vector[float64]) vector.Vector[float64] {
	return l.activation.Activate(input)
}
func (l Layer) derivative(input vector.Vector[float64]) vector.Vector[float64] {
	return l.activation.Derivative(input)
}

// forwardValues allows to pass forward input vector and, using it, calculate
// its activation vector (a) and output vector (z). Store calculated values
// in cache structure for farther needs (local loss calculations).
//
// - input, hidden layers -> respond with output vector
//
// - output layer		  -> respond with activation vector (for farther percent calculation and evaluation)
func (l *Layer) forwardValues(inputs vector.Vector[float64]) vector.Vector[float64] {

	// 1. Calculate sum vector using neurons' weights and input vector
	sumVector := make(vector.Vector[float64], len(l.neurons))
	for i, neuron := range l.neurons {
		sumVector[i] = neuron.Sum(inputs)
	}

	// 2. Receive activation vector using activation function (sigmoid, softmax, etc.)
	l.cache.activation = l.activate(sumVector)
	l.cache.output = l.cache.activation

	// 3. Get derivative vector, that will be needed for local loss func
	l.cache.derivative = l.derivative(l.cache.output)

	return l.cache.activation
}

func (l Layer) cleanCache() {
	l.cache.clean()
}

// calcWeightLossSum calculates a dot product of
//
// * local loss vector
//
// * weight column vector (vector as a column of weights with specific index)
func (l Layer) calcWeightLossSum(lossVector vector.Vector[float64], index int) float64 {

	// Make weight vector as a column with specific index (take 1 element by index from each neuron's weight)
	weightColumnVector := make(vector.Vector[float64], len(l.neurons))
	for i, neuron := range l.neurons {
		weightColumnVector[i] = neuron.Weight()[index]
	}

	// Make a dot product
	return weightColumnVector.Dot(lossVector)
}

func (l Layer) Derivative(index int) float64 {
	return l.cache.derivative[index]
}

func (l Layer) addBatchLoss(lossVector, inputs vector.Vector[float64]) {
	for i, neuron := range l.neurons {

		deltaW := inputs.Scl(lossVector[i])
		deltaB := lossVector[i]

		neuron.AddBatchLoss(deltaW, deltaB)

		// fmt.Println("Layer local loss:", deltaW, deltaB)
	}
}

func (l Layer) applyLoss() {
	for _, neuron := range l.neurons {
		neuron.ApplyLoss()
	}
}

// layerFromNeurons rebuilds a layer around neurons that already carry their
// trained weights, used when loading a saved model.
func layerFromNeurons(neurons []*neuron.Neuron, activation activationFunc) *Layer {
	return &Layer{
		neurons:    neurons,
		activation: activation,
		cache:      newLayerCache(len(neurons)),
	}
}

// activationByName maps the name stored in a saved model back to the
// implementation. Saving the name rather than anything structural is what
// keeps a model file readable and lets a human see which activations a
// network was built from.
func activationByName(name string) (activationFunc, error) {
	for _, candidate := range []activationFunc{
		activation.SigmoidFunc{},
		activation.SoftMaxFunc{},
	} {
		if candidate.Name() == name {
			return candidate, nil
		}
	}
	return nil, fmt.Errorf("%w: %q", ErrUndefinedActivationFunc, name)
}
