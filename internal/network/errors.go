package network

import "errors"

var (
	ErrModelWithoutLayers       error = errors.New("mlp model error: model has not enough layers (must be >= 2)")
	ErrModelTrainWithoutDataset error = errors.New("mlp model error: dataset is not provided for model training")
)
