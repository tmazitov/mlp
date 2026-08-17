package analytics

import "mlp/pkg/vector"

// Standardize returns a new Dataset where every feature column has been
// replaced by its z-score, (x-mean)/stddev, with mean/stddev computed per
// column from this dataset's own rows. d itself is left untouched.
//
// It works on whatever columns are present in d.Rows[0].Features, so it's
// meant to run after ExtractFields has narrowed the dataset down to the
// fields actually used for training.
func (d *Dataset) Standardize() *Dataset {
	if len(d.Rows) == 0 {
		return &Dataset{}
	}

	featureCount := len(d.Rows[0].Features)

	columns := make([][]float64, featureCount)
	for _, row := range d.Rows {
		for j, v := range row.Features {
			columns[j] = append(columns[j], v)
		}
	}

	stats := make([]FeatureStats, featureCount)
	for j, values := range columns {
		stats[j] = computeStats("", values)
	}

	rows := make([]Row, len(d.Rows))
	for i, row := range d.Rows {
		features := make(vector.Vector[float64], featureCount)
		for j, v := range row.Features {
			stddev := stats[j].StdDev
			if stddev == 0 {
				stddev = 1 // constant column: leave it centered at 0 instead of dividing by zero
			}
			features[j] = (v - stats[j].Mean) / stddev
		}

		rows[i] = Row{
			Diagnosis:        row.Diagnosis,
			Features:         features,
			extractedIndexes: row.extractedIndexes,
		}
	}

	return &Dataset{Rows: rows}
}
