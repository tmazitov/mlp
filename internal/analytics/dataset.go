// Package analytics loads and summarizes the WDBC breast cancer dataset
// (data.csv): 1 id column, 1 diagnosis column (M/B), and 30 numeric
// features — 10 measurements reported as mean, standard error, and worst.
package analytics

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

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

type Row struct {
	Diagnosis string
	Features  []float64
}

type Dataset struct {
	Rows []Row
}

func Load(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open dataset: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 2 + len(FeatureNames)

	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read dataset: %w", err)
	}

	ds := &Dataset{Rows: make([]Row, 0, len(records))}
	for i, record := range records {
		features := make([]float64, len(FeatureNames))
		for j := range FeatureNames {
			v, err := strconv.ParseFloat(record[2+j], 64)
			if err != nil {
				return nil, fmt.Errorf("row %d: feature %s: %w", i, FeatureNames[j], err)
			}
			features[j] = v
		}

		ds.Rows = append(ds.Rows, Row{
			Diagnosis: record[1],
			Features:  features,
		})
	}

	return ds, nil
}

// ClassCounts returns the number of rows per diagnosis class.
func (d *Dataset) ClassCounts() map[string]int {
	counts := map[string]int{}
	for _, row := range d.Rows {
		counts[row.Diagnosis]++
	}
	return counts
}
