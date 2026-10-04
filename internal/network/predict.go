package network

import (
	"mlp/internal/analytics"
	"mlp/pkg/vector"
)

// Evaluation is the result of running a trained model over a labelled set.
//
// Malignant is treated as the positive class: in this dataset a false
// negative is a malignant tumour reported as benign, which is a far more
// costly mistake than the reverse, and plain accuracy hides that asymmetry.
type Evaluation struct {
	Rows     int
	Loss     float64
	Accuracy float64

	TruePositives  int
	TrueNegatives  int
	FalsePositives int
	FalseNegatives int
}

// Evaluate prepares dataset exactly the way training prepared its own rows
// — the stored fields, in the stored order, scaled with the stored
// parameters — then predicts every row and scores the result.
//
// Doing the preparation here, from the values saved in the model file, is
// the whole reason those values are saved: the caller cannot get it wrong.
func (t *TrainedModel) Evaluate(dataset *analytics.Dataset) (Evaluation, error) {
	var result Evaluation

	if t.Network == nil || len(t.Network.layers) == 0 {
		return result, ErrModelWithoutLayers
	}
	if dataset == nil || len(dataset.Rows) == 0 {
		return result, ErrModelTrainWithoutDataset
	}

	prepared := t.Scaler.Apply(dataset.ExtractFields(t.Fields...))

	var lossSum float64
	for _, row := range prepared.Rows {
		probabilities := t.Network.forward(row.Features)
		answer := row.DiagnosisVector()

		// The subject writes the loss for this phase in its binary form,
		// -1/N sum[y ln p + (1-y) ln(1-p)]. With two softmax outputs, whose
		// probabilities sum to 1, that is the same number as the
		// categorical form used during training, so the curves and this
		// score stay on one scale.
		lossSum += crossEntropyLoss(answer, probabilities)

		predicted := classFromIndex(probabilities.ArgMax())
		actual := classFromIndex(answer.ArgMax())

		switch {
		case predicted == MalignantClass && actual == MalignantClass:
			result.TruePositives++
		case predicted == BenignClass && actual == BenignClass:
			result.TrueNegatives++
		case predicted == MalignantClass && actual == BenignClass:
			result.FalsePositives++
		default:
			result.FalseNegatives++
		}
	}

	result.Rows = len(prepared.Rows)
	result.Loss = lossSum / float64(result.Rows)
	result.Accuracy = float64(result.TruePositives+result.TrueNegatives) / float64(result.Rows)

	return result, nil
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
