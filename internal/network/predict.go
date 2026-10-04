package network

import (
	"mlp/internal/analytics"
	"mlp/internal/analytics/log"
	"mlp/pkg/vector"
)

// Evaluation is what prediction reports. It is the same measurement the
// training loop records each epoch, so a prediction score and the last
// point on a learning curve are directly comparable.
type Evaluation = log.Metrics

// Evaluate prepares dataset exactly the way training prepared its own rows
// — the stored fields, in the stored order, scaled with the stored
// parameters — then predicts every row and scores the result.
//
// Doing the preparation here, from the values saved in the model file, is
// the whole reason those values are saved: the caller cannot get it wrong.
//
// The subject writes this phase's loss in its binary form,
// -1/N sum[y ln p + (1-y) ln(1-p)]. With two softmax outputs, whose
// probabilities sum to 1, that is the same number as the categorical form
// used during training.
func (t *TrainedModel) Evaluate(dataset *analytics.Dataset) (Evaluation, error) {
	var result Evaluation

	if t.Network == nil || len(t.Network.layers) == 0 {
		return result, ErrModelWithoutLayers
	}
	if dataset == nil || len(dataset.Rows) == 0 {
		return result, ErrModelTrainWithoutDataset
	}

	return t.Network.score(t.Scaler.Apply(dataset.ExtractFields(t.Fields...))), nil
}

// Predict returns the class the network assigns to one input vector that
// has already been extracted and scaled — see TrainedModel.Evaluate for the
// path that does that preparation for you.
func (m MLP) Predict(inputs vector.Vector[float64]) predictedClass {
	return classFromIndex(m.forward(inputs).ArgMax())
}

// forward runs one input through every layer and returns the output
// layer's probabilities.
func (m MLP) forward(inputs vector.Vector[float64]) vector.Vector[float64] {
	activations := inputs
	for _, layer := range m.layers {
		activations = layer.forwardValues(activations)
	}
	return activations
}
