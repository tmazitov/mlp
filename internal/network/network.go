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

func (m MLP) Train(dataset *analytics.Dataset) error {

	if dataset == nil {
		return ErrModelTrainWithoutDataset
	}

	if len(m.layers) < 2 {
		return ErrModelWithoutLayers
	}

	reader := dataset.NewReader()

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

		lossValues := []float64{}

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

				// Save loss value for statistics
				lossValues = append(lossValues, float64(layerLossVectors[len(m.layers)-1].NormInf()))

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

				// Apply local loss value for each layer
				for l, layer := range m.layers {
					if layerLossVectors[l] == nil {
						continue
					}
					layer.applyLoss(layerLossVectors[l], inputVectors[l])
					layer.cleanCache()
				}

				clear(inputVectors)
				clear(layerLossVectors)
			}

		}
		// Log the epoch's result using specific chanel if its proceeded
		if m.config.LogsChan != nil {
			m.config.LogsChan <- log.NewTrainingStat(epoch, lossValues)
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
