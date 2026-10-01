package analytics

import "mlp/pkg/vector"

// FeatureScaler holds the per-feature mean/stddev fitted from one dataset,
// so the exact same z-score transform can be reapplied to another dataset
// (e.g. a validation split, or a future prediction input) instead of being
// recomputed from different data — mixing the two would leak validation
// statistics into training.
type FeatureScaler struct {
	mean   []float64
	stddev []float64
}

// Fit computes a FeatureScaler from d's own rows: mean/stddev per column.
// It works on whatever columns are present in d.Rows[0].Features, so it's
// meant to run after ExtractFields has narrowed the dataset down to the
// fields actually used for training.
func (d *Dataset) Fit() FeatureScaler {
	if len(d.Rows) == 0 {
		return FeatureScaler{}
	}

	featureCount := len(d.Rows[0].Features)

	columns := make([][]float64, featureCount)
	for _, row := range d.Rows {
		for j, v := range row.Features {
			columns[j] = append(columns[j], v)
		}
	}

	mean := make([]float64, featureCount)
	stddev := make([]float64, featureCount)
	for j, values := range columns {
		stats := computeStats("", values)
		mean[j] = stats.Mean
		stddev[j] = stats.StdDev
		if stddev[j] == 0 {
			stddev[j] = 1 // constant column: leave it centered at 0 instead of dividing by zero
		}
	}

	return FeatureScaler{mean: mean, stddev: stddev}
}

// Apply returns a new Dataset with every value replaced by its z-score
// under s, (x-mean)/stddev. d itself is left untouched.
func (s FeatureScaler) Apply(d *Dataset) *Dataset {
	rows := make([]Row, len(d.Rows))
	for i, row := range d.Rows {
		features := make(vector.Vector[float64], len(row.Features))
		for j, v := range row.Features {
			features[j] = (v - s.mean[j]) / s.stddev[j]
		}

		rows[i] = Row{
			ID:               row.ID,
			Diagnosis:        row.Diagnosis,
			Features:         features,
			extractedIndexes: row.extractedIndexes,
		}
	}

	return &Dataset{Rows: rows}
}

// Standardize is Fit+Apply on the same dataset: mean/stddev are computed
// from d's own rows and immediately applied to d itself. d is left
// untouched; a new Dataset is returned.
func (d *Dataset) Standardize() *Dataset {
	return d.Fit().Apply(d)
}
