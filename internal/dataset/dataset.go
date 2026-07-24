package dataset

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

type Dataset []*Record

func NewDataset(csvFilePath string) (*Dataset, error) {

	data, err := extractData(csvFilePath)
	if err != nil {
		return nil, fmt.Errorf("dataset parse error: %w", err)
	}

	return (*Dataset)(&data), nil
}

func extractData(csvFilePath string) ([]*Record, error) {

	// Open the file
	file, err := os.Open(csvFilePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	records := []*Record{}

	reader := csv.NewReader(file)
	counter := 0
	for {

		// Read record
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("dataset record %d error: %w", counter, err)
		}

		if len(record) < 3 {
			return nil, fmt.Errorf("dataset record %d error: has not enough parameters (%d)", counter, len(records))
		}

		// Extract params
		id, err := strconv.Atoi(record[0])
		if err != nil {
			return nil, fmt.Errorf("dataset record %d error: %w", counter, err)
		}
		name := record[1]
		features := make([]float64, 0, len(record[1:]))
		for _, featureRaw := range record[1:] {
			value, err := strconv.ParseFloat(featureRaw, 10)
			if err != nil {
				return nil, fmt.Errorf("dataset record %d error: %w", counter, err)
			}
			features = append(features, value)
		}

		// Gather all
		records = append(records, NewRecord(id, name, features))

		counter += 1
	}

	return records, nil
}
