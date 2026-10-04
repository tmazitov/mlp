package network

import (
	"mlp/internal/analytics"
	"mlp/internal/analytics/log"
	"mlp/pkg/vector"
)

// score runs a forward-only pass (no weight updates) over dataset and
// tallies loss and the confusion matrix.
//
// Training calls it once per epoch on the held-out validation set, so it
// never sees the rows it is scored against, and prediction calls it on
// whatever set it was given. One implementation means the number a run
// reports and the number a prediction reports are the same measurement.
func (m MLP) score(dataset *analytics.Dataset) log.Metrics {
	var metrics log.Metrics

	if dataset == nil || len(dataset.Rows) == 0 {
		return metrics
	}

	for _, row := range dataset.Rows {
		probabilities := m.forward(row.Features)
		answer := row.DiagnosisVector()

		metrics.Loss += crossEntropyLoss(answer, probabilities)
		recordPrediction(&metrics, probabilities, answer)
	}

	metrics.Loss /= float64(metrics.Rows)
	return metrics
}

// recordPrediction folds one row's outcome into metrics. Both the training
// loop and score go through here so the positive class is decided in one
// place.
func recordPrediction(metrics *log.Metrics, probabilities, answer vector.Vector[float64]) {
	predicted := classFromIndex(probabilities.ArgMax())
	actual := classFromIndex(answer.ArgMax())

	metrics.Add(predicted == MalignantClass, actual == MalignantClass)
}
