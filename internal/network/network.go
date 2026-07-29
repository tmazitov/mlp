package network

import (
	"io"
	"mlp/internal/analytics"
	"mlp/pkg/vector"
)

type MLPConfig struct {
	Epochs         int
	BatchSize      int
	LossFunc       lossFunc
	Mode           modelMode
	LogsChan       chan string
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

func (m MLP) AddLayer(neuronCount uint, activation activationFunc) {
	m.layers = append(m.layers, NewLayer(neuronCount, activation))
}

func (m MLP) Train(dataset *analytics.Dataset) error {

	if dataset == nil {
		return ErrModelTrainWithoutDataset
	}

	if len(m.layers) < 2 {
		return ErrModelWithoutLayers
	}

	reader := dataset.NewReader()

	for epoch := range m.config.Epochs {
		_ = epoch
		for {
			batch, err := reader.Read(m.config.BatchSize)
			if err == io.EOF {
				break
			} else if err != nil {
				return err
			}

			for _, row := range batch {

				v := row.Features

				// Forward values from layer to layer
				for _, layer := range m.layers {
					v = layer.forwardValues(v)
				}

				// Calculate the total loss value for this row
				// Apply loss value for each layer
			}

		}
		// Log the epoch's result using specific chanel if its proceeded
	}
	// Save weights to the file

	return nil
}

func (m MLP) Predict(inputs vector.Vector[float64]) predictedClass {
	return BenignClass
}
