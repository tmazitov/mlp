package network

import "mlp/internal/analytics"

// evaluate runs a forward-only pass (no weight updates) over dataset and
// returns the average cross-entropy loss (the same metric used during
// training — the two must match or the curves are not comparable) and
// classification accuracy. Used once per epoch on the held-out
// validation set, so it never sees the rows it's scored against.
func (m MLP) evaluate(dataset *analytics.Dataset) (avgLoss, accuracy float64) {
	if dataset == nil || len(dataset.Rows) == 0 {
		return 0, 0
	}

	var lossSum, correct float64

	for _, row := range dataset.Rows {
		activations := m.forward(row.Features)
		answer := row.DiagnosisVector()
		lossSum += crossEntropyLoss(answer, activations)
		if activations.ArgMax() == answer.ArgMax() {
			correct++
		}
	}

	n := float64(len(dataset.Rows))
	return lossSum / n, correct / n
}
