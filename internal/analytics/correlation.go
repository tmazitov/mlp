package analytics

import (
	"math"
	"sort"
)

type CorrelationPair struct {
	FeatureA    string
	FeatureB    string
	Correlation float64
}

// CorrelationMatrix returns the full, symmetric NxN Pearson correlation
// matrix across all features, in FeatureNames order. The diagonal is 1.
func (d *Dataset) CorrelationMatrix() [][]float64 {
	columns := make([][]float64, len(FeatureNames))
	for i := range columns {
		columns[i] = make([]float64, len(d.Rows))
	}
	for r, row := range d.Rows {
		for i, v := range row.Features {
			columns[i][r] = v
		}
	}

	n := len(FeatureNames)
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
		matrix[i][i] = 1
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			c := pearson(columns[i], columns[j])
			matrix[i][j] = c
			matrix[j][i] = c
		}
	}
	return matrix
}

// TopCorrelations returns feature pairs whose absolute Pearson correlation
// is at least minAbs, sorted from strongest to weakest. Useful for spotting
// redundant features (e.g. a *_mean and its *_worst counterpart).
func (d *Dataset) TopCorrelations(minAbs float64) []CorrelationPair {
	matrix := d.CorrelationMatrix()

	var pairs []CorrelationPair
	for i := range FeatureNames {
		for j := i + 1; j < len(FeatureNames); j++ {
			c := matrix[i][j]
			if math.Abs(c) >= minAbs {
				pairs = append(pairs, CorrelationPair{
					FeatureA:    FeatureNames[i],
					FeatureB:    FeatureNames[j],
					Correlation: c,
				})
			}
		}
	}

	sort.Slice(pairs, func(i, j int) bool {
		return math.Abs(pairs[i].Correlation) > math.Abs(pairs[j].Correlation)
	})

	return pairs
}

func pearson(a, b []float64) float64 {
	n := float64(len(a))

	sumA, sumB := 0.0, 0.0
	for i := range a {
		sumA += a[i]
		sumB += b[i]
	}
	meanA, meanB := sumA/n, sumB/n

	cov, varA, varB := 0.0, 0.0, 0.0
	for i := range a {
		da, db := a[i]-meanA, b[i]-meanB
		cov += da * db
		varA += da * da
		varB += db * db
	}

	denom := math.Sqrt(varA * varB)
	if denom == 0 {
		return 0
	}
	return cov / denom
}
