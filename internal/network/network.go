package network

import (
	"io"
	"mlp/internal/analytics"
	"mlp/internal/analytics/log"
	"mlp/internal/network/neuron"
	"mlp/pkg/vector"
)

type MLPConfig struct {
	Epochs         int
	BatchSize      int
	LossFunc       lossFunc
	Mode           modelMode
	LogsChan       chan log.TrainingStat
	WeightFilePath string
}

type MLP struct {
	layers []*Layer
	config MLPConfig
}

func NewMLP(config MLPConfig) *MLP {

	model := &MLP{
		config: config,
	}

	return model
}

func (m *MLP) AddLayer(neuronCount uint, activation activationFunc, neuronParams neuron.NeuronParams) error {

	layer, err := NewLayer(neuronCount, activation, neuronParams)
	if err != nil {
		return err
	}

	m.layers = append(m.layers, layer)
	return nil
}

func (m MLP) Train(trainSet, valSet *analytics.Dataset) error {

	if trainSet == nil {
		return ErrModelTrainWithoutDataset
	}

	if len(m.layers) < 2 {
		return ErrModelWithoutLayers
	}

	if m.config.LossFunc != CrossEntropyLossFunc {
		return ErrUnsupportedLossFunc
	}

	reader := trainSet.NewReader()

	// General explanation of training:
	//
	//
	// input_layer (1) <--> hidden_layers (n) <--> output_layer(1)
	//
	//
	// 1. Pass forward the record data from layer to layer,
	// recording the current output vector as an input of the next layer.
	//
	// 2. Last input vector implies the model's prediction, that should be
	// evaluated by converted to first loss vector.
	//
	// 3. Using back propagation algorithm, calculate local loss vector
	// other layers until that time when it reaches the input_layer
	//
	// 4. Apply loss vectors for corresponding layers to modify wights and bias.

	// The slice of input vectors. One slot for the row's raw features plus
	// one for every layer's output (inputVectors[l] is layer l's input).
	inputVectors := make([]vector.Vector[float64], len(m.layers)+1)

	// The slice of loss vectors that dedicated for tuning of layers' weights.
	// Every layer l(i) after 3-rd stem has it's own loss vector δ(i)
	layerLossVectors := make([]vector.Vector[float64], len(m.layers))

	for epoch := range m.config.Epochs {

		var trainLossSum, trainCorrect float64
		var trainCount int

		for {
			batch, err := reader.Read(m.config.BatchSize)
			if err == io.EOF {
				break
			} else if err != nil {
				return err
			}

			for _, row := range batch {

				// First input vector
				inputVectors[0] = row.Features

				// Forward row.Features vector from layer to layer,
				// where output of the current level will be an input for the next one.
				// l1 -z1-> | -a2-> l2 -z2-> | -a3-> l3 -z3-> | ...
				// a - input vectors, z - output vectors
				for i, layer := range m.layers {
					output := layer.forwardValues(inputVectors[i])
					inputVectors[i+1] = output
				}

				// Output of the last layer is a prediction vector,
				// where each value is a probability of belonging to specific class.
				predict := inputVectors[len(inputVectors)-1]
				answerVector := row.DiagnosisVector()

				// Calculation of loss value is simplified due to
				// usage of 2 algorithms together: Softmax + Cross Entropy
				layerLossVectors[len(m.layers)-1] = predict.Sub(answerVector)

				// Save loss/accuracy for this epoch's training statistics.
				// This is the reported metric; the gradient above is the
				// softmax+cross-entropy shortcut, which is why cross-entropy
				// is the only loss Train accepts.
				trainLossSum += crossEntropyLoss(answerVector, predict)
				if predict.ArgMax() == answerVector.ArgMax() {
					trainCorrect++
				}
				trainCount++

				// Backward loop move through layers to calculate local loss.
				// It starts from last hidden layer.
				for l := len(m.layers) - 2; l >= 0; l-- {

					currentLayer := m.layers[l]
					nextLayer := m.layers[l+1]

					lastLossVector := layerLossVectors[l+1]

					lossVector := make(vector.Vector[float64], len(currentLayer.neurons))
					for i := range lossVector {

						weightLossSum := nextLayer.calcWeightLossSum(lastLossVector, i)

						derivative := currentLayer.Derivative(i)

						lossVector[i] = weightLossSum * derivative
					}

					layerLossVectors[l] = lossVector
				}

				// Accumulate this row's gradient for every layer.
				//
				// Caches must NOT be cleaned inside this loop: inputVectors[l+1]
				// is the very slice layer l stored as cache.activation (see
				// forwardValues), so cleaning layer l here would zero the inputs
				// layer l+1 is about to multiply its delta by — leaving every
				// layer after the first with a zero weight gradient.
				for l, layer := range m.layers {
					if layerLossVectors[l] == nil {
						continue
					}
					layer.addBatchLoss(layerLossVectors[l], inputVectors[l])
				}

				for _, layer := range m.layers {
					layer.cleanCache()
				}

				clear(inputVectors)
				clear(layerLossVectors)
			}

			for _, layer := range m.layers {
				layer.applyLoss()
			}

		}
		// Evaluate on the held-out validation set — a forward-only pass, so
		// it never influences the weights, only the reported metrics.
		valLoss, valAccuracy := m.evaluate(valSet)

		// Log the epoch's result using specific chanel if its proceeded
		if m.config.LogsChan != nil {
			m.config.LogsChan <- log.TrainingStat{
				Epoch:         epoch,
				TrainLoss:     trainLossSum / float64(trainCount),
				ValLoss:       valLoss,
				TrainAccuracy: trainCorrect / float64(trainCount),
				ValAccuracy:   valAccuracy,
			}
		}

		// Start batch reading from the beginning
		reader.Reset()
	}
	// Save weights to the file

	return nil
}

// func (m MLP) calculateInnerLoss(lastLoss vector.Vector[float64]) vector.Vector[float64] {

// }

func (m MLP) Predict(inputs vector.Vector[float64]) predictedClass {
	return BenignClass
}
