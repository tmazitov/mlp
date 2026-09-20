package network

import "mlp/internal/analytics"

// evaluate runs a forward-only pass (no weight updates) over dataset and
// returns the average loss (NormInf of predict-answer, same metric used
// during training) and classification accuracy. Used once per epoch on the
// held-out validation set, so it never sees the rows it's scored against.
func (m MLP) evaluate(dataset *analytics.Dataset) (avgLoss, accuracy float64) {
	if dataset == nil || len(dataset.Rows) == 0 {
		return 0, 0
	}

	var lossSum, correct float64

	for _, row := range dataset.Rows {
		activations := row.Features
		for _, layer := range m.layers {
			activations = layer.forwardValues(activations)
		}

		answer := row.DiagnosisVector()
		lossSum += float64(activations.Sub(answer).NormInf())
		if activations.ArgMax() == answer.ArgMax() {
			correct++
		}
	}

	n := float64(len(dataset.Rows))
	return lossSum / n, correct / n
}
