package analytics

import "math"

type FeatureStats struct {
	Name   string
	Mean   float64
	StdDev float64
	Min    float64
	Max    float64
}

func computeStats(name string, values []float64) FeatureStats {
	n := float64(len(values))

	sum, min, max := 0.0, math.Inf(1), math.Inf(-1)
	for _, v := range values {
		sum += v
		min = math.Min(min, v)
		max = math.Max(max, v)
	}
	mean := sum / n

	variance := 0.0
	for _, v := range values {
		d := v - mean
		variance += d * d
	}
	variance /= n

	return FeatureStats{
		Name:   name,
		Mean:   mean,
		StdDev: math.Sqrt(variance),
		Min:    min,
		Max:    max,
	}
}

// FeatureStats computes overall per-feature statistics across all rows,
// regardless of diagnosis, in FeatureNames order.
func (d *Dataset) FeatureStats() []FeatureStats {
	values := make([][]float64, len(FeatureNames))
	for _, row := range d.Rows {
		for i, v := range row.Features {
			values[i] = append(values[i], v)
		}
	}

	stats := make([]FeatureStats, len(FeatureNames))
	for i, name := range FeatureNames {
		stats[i] = computeStats(name, values[i])
	}
	return stats
}

// ValuesByDiagnosis returns the raw values of a single feature (by index
// into FeatureNames), grouped by diagnosis class.
func (d *Dataset) ValuesByDiagnosis(featureIndex int) map[string][]float64 {
	result := map[string][]float64{}
	for _, row := range d.Rows {
		result[row.Diagnosis] = append(result[row.Diagnosis], row.Features[featureIndex])
	}
	return result
}

// FeatureStatsByDiagnosis computes per-feature statistics separately for
// each diagnosis class (e.g. "B" and "M"), in FeatureNames order.
func (d *Dataset) FeatureStatsByDiagnosis() map[string][]FeatureStats {
	valuesByClass := map[string][][]float64{}
	for _, row := range d.Rows {
		values, ok := valuesByClass[row.Diagnosis]
		if !ok {
			values = make([][]float64, len(FeatureNames))
		}
		for i, v := range row.Features {
			values[i] = append(values[i], v)
		}
		valuesByClass[row.Diagnosis] = values
	}

	result := map[string][]FeatureStats{}
	for diagnosis, values := range valuesByClass {
		stats := make([]FeatureStats, len(FeatureNames))
		for i, name := range FeatureNames {
			stats[i] = computeStats(name, values[i])
		}
		result[diagnosis] = stats
	}

	return result
}
