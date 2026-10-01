package network

import (
	"encoding/json"
	"fmt"
	"os"

	"mlp/internal/analytics"
	"mlp/internal/network/neuron"
	"mlp/pkg/vector"
)

// TrainedModel is everything prediction needs to reproduce what training
// produced: the network itself, the feature columns it was trained on, and
// the scaler those columns were standardised with.
//
// The last two matter as much as the weights. Feed a prediction raw,
// unscaled columns — or the right columns in the wrong order — and the
// network sees inputs from a different distribution than the one it
// learned, so its answers are meaningless rather than merely wrong.
type TrainedModel struct {
	Network *MLP
	Fields  []string
	Scaler  analytics.FeatureScaler
}

// savedModel is the on-disk shape. It is kept separate from TrainedModel so
// the file format stays explicit and readable rather than following
// whatever the in-memory types happen to look like.
type savedModel struct {
	Fields []string     `json:"fields"`
	Scaler savedScaler  `json:"scaler"`
	Layers []savedLayer `json:"layers"`
}

type savedScaler struct {
	Mean   []float64 `json:"mean"`
	StdDev []float64 `json:"stddev"`
}

type savedLayer struct {
	Activation string        `json:"activation"`
	Neurons    []savedNeuron `json:"neurons"`
}

type savedNeuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

// Save writes the model to path as JSON: topology, weights and biases, plus
// the fields and scaler parameters needed to prepare an input the same way
// training did.
func (t *TrainedModel) Save(path string) error {
	if t.Network == nil || len(t.Network.layers) == 0 {
		return ErrModelWithoutLayers
	}

	saved := savedModel{
		Fields: t.Fields,
		Scaler: savedScaler{
			Mean:   t.Scaler.Mean(),
			StdDev: t.Scaler.StdDev(),
		},
		Layers: make([]savedLayer, 0, len(t.Network.layers)),
	}

	for _, layer := range t.Network.layers {
		neurons := make([]savedNeuron, 0, len(layer.neurons))
		for _, n := range layer.neurons {
			neurons = append(neurons, savedNeuron{
				Weights: n.Weight(),
				Bias:    n.Bias(),
			})
		}

		saved.Layers = append(saved.Layers, savedLayer{
			Activation: layer.activation.Name(),
			Neurons:    neurons,
		})
	}

	// Indented, because a model file an evaluator can open and read is
	// worth more here than a few saved bytes.
	content, err := json.MarshalIndent(saved, "", "  ")
	if err != nil {
		return fmt.Errorf("encode model: %w", err)
	}

	if err := os.WriteFile(path, content, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}

	return nil
}

// LoadModel reads back a model written by Save. The returned network is
// ready to predict: every neuron carries its trained weights, so nothing is
// re-initialised randomly.
func LoadModel(path string) (*TrainedModel, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}

	var saved savedModel
	if err := json.Unmarshal(content, &saved); err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}

	if len(saved.Layers) == 0 {
		return nil, ErrModelWithoutLayers
	}

	model := &MLP{layers: make([]*Layer, 0, len(saved.Layers))}

	for _, savedLayer := range saved.Layers {
		activationFn, err := activationByName(savedLayer.Activation)
		if err != nil {
			return nil, err
		}

		neurons := make([]*neuron.Neuron, 0, len(savedLayer.Neurons))
		for i, savedNeuron := range savedLayer.Neurons {
			neurons = append(neurons, neuron.FromWeights(
				uint(i), vector.Vector[float64](savedNeuron.Weights), savedNeuron.Bias))
		}

		model.layers = append(model.layers, layerFromNeurons(neurons, activationFn))
	}

	return &TrainedModel{
		Network: model,
		Fields:  saved.Fields,
		Scaler:  analytics.NewFeatureScaler(saved.Scaler.Mean, saved.Scaler.StdDev),
	}, nil
}
