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

func (d *Dataset) NewReader() *batchReader {
	return newBatchReader(d)
}

func (d Dataset) ExtractFields(fieldNames ...string) Dataset {
	newDataset := Dataset{
		Rows: make([]Row, 0, len(d.Rows)),
	}

	for _, row := range d.Rows {
		newDataset.Rows = append(newDataset.Rows, row.ExtractFields(fieldNames...))
	}

	return newDataset
}

// ClassCounts returns the number of rows per diagnosis class.
func (d *Dataset) ClassCounts() map[string]int {
	counts := map[string]int{}
	for _, row := range d.Rows {
		counts[row.Diagnosis]++
	}
	return counts
}
