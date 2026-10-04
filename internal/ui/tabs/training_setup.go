package tabs

import (
	"fmt"

	"mlp/internal/analytics"
)

// fieldsToTrain are the columns the network is trained on: one size
// measurement standing in for the whole radius/perimeter/area cluster,
// which is near-duplicated information, plus the shape and texture
// measurements that separate the classes on their own.
var fieldsToTrain = []string{
	"area_worst",
	"concave_points_worst",
	"concavity_mean",
	"texture_worst",
	"smoothness_worst",
	"symmetry_worst",
	"fractal_dimension_worst",
	"compactness_worst",
}

// trainingData is a prepared pair of sets plus the scaler they were
// standardised with, which the model has to be saved alongside.
type trainingData struct {
	train  *analytics.Dataset
	val    *analytics.Dataset
	scaler analytics.FeatureScaler
}

// loadTrainingData reads the two files the split phase wrote, narrows them
// to the training columns and standardises them.
//
// The sets come from those files rather than from re-dividing data.csv,
// otherwise the split phase would have no effect on what is trained. Both
// the single-run and the comparison paths go through here so they cannot
// drift apart in how they prepare their input.
func loadTrainingData() (trainingData, error) {
	var data trainingData

	train, err := analytics.Load(trainingCSVPath)
	if err != nil {
		return data, fmt.Errorf("load %s: %w — run the Split tab first", trainingCSVPath, err)
	}

	val, err := analytics.Load(validationCSVPath)
	if err != nil {
		return data, fmt.Errorf("load %s: %w — run the Split tab first", validationCSVPath, err)
	}

	train = train.ExtractFields(fieldsToTrain...)
	val = val.ExtractFields(fieldsToTrain...)

	// mean/stddev are fit on the training rows only, then reapplied to
	// validation — fitting on the combined set would leak validation
	// statistics into training.
	scaler := train.Fit()

	data.train = scaler.Apply(train)
	data.val = scaler.Apply(val)
	data.scaler = scaler

	return data, nil
}
