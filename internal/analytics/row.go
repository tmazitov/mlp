package analytics

import (
	"mlp/pkg/vector"
	"slices"
)

type Row struct {
	Diagnosis        string
	Features         vector.Vector[float64]
	extractedIndexes []int
}

func (r Row) ExtractFields(fieldNames ...string) Row {

	indexMap := indexNames(FeatureNames)

	newRow := Row{
		Diagnosis: r.Diagnosis,
		Features:  make([]float64, 0, len(fieldNames)),
	}

	for _, field := range fieldNames {
		index := indexMap[field]
		if len(r.extractedIndexes) != 0 && slices.Index(r.extractedIndexes, index) == -1 {
			continue
		}

		newRow.Features = append(newRow.Features, r.Features[index])
		newRow.extractedIndexes = append(newRow.extractedIndexes, index)
	}

	return newRow
}

var FeatureNames = []string{
	"radius_mean", "texture_mean", "perimeter_mean", "area_mean",
	"smoothness_mean", "compactness_mean", "concavity_mean",
	"concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
	"radius_se", "texture_se", "perimeter_se", "area_se",
	"smoothness_se", "compactness_se", "concavity_se",
	"concave_points_se", "symmetry_se", "fractal_dimension_se",
	"radius_worst", "texture_worst", "perimeter_worst", "area_worst",
	"smoothness_worst", "compactness_worst", "concavity_worst",
	"concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
}

// featureIndex maps a feature name to its position in FeatureNames (and
// therefore in Row.Features), so callers like ExtractFields can look up a
// field by name in O(1) instead of scanning the slice.
func indexNames(names []string) map[string]int {
	index := make(map[string]int, len(names))
	for i, name := range names {
		index[name] = i
	}
	return index
}
