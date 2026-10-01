package analytics

import (
	"encoding/csv"
	"fmt"
	"math/rand/v2"
	"os"
	"strconv"
)

// Split partitions d's rows into two new datasets: trainRatio of the rows
// for training, the rest for validation. d itself is left untouched.
//
// When shuffle is true the rows are randomised first, which is what you
// normally want — a dataset ordered by class (or by anything correlated
// with it) would otherwise produce a lopsided split. Passing false keeps
// the file's own order.
//
// seed makes a shuffled split reproducible: the same seed over the same
// input always deals the same two sets. Pass nil for a different draw every
// run. It has no effect when shuffle is false, since nothing is random then.
func (d *Dataset) Split(trainRatio float64, shuffle bool, seed *uint64) (train, val *Dataset) {
	rows := make([]Row, len(d.Rows))
	copy(rows, d.Rows)

	if shuffle {
		swap := func(i, j int) { rows[i], rows[j] = rows[j], rows[i] }
		if seed != nil {
			rand.New(rand.NewPCG(*seed, *seed)).Shuffle(len(rows), swap)
		} else {
			rand.Shuffle(len(rows), swap)
		}
	}

	cut := int(float64(len(rows)) * trainRatio)

	trainRows := make([]Row, cut)
	copy(trainRows, rows[:cut])

	valRows := make([]Row, len(rows)-cut)
	copy(valRows, rows[cut:])

	return &Dataset{Rows: trainRows}, &Dataset{Rows: valRows}
}

// WriteCSV writes d back out in the same shape Load reads: one row per
// line, starting with the id and the diagnosis, followed by the feature
// columns. Keeping the layout identical is what lets the files produced by
// a split be fed straight back into Load.
func (d *Dataset) WriteCSV(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	writer := csv.NewWriter(f)

	for _, row := range d.Rows {
		record := make([]string, 0, 2+len(row.Features))
		record = append(record, row.ID, row.Diagnosis)
		for _, value := range row.Features {
			// 'g' with precision -1 round-trips the float through
			// strconv.ParseFloat without widening 1001 into 1001.0000.
			record = append(record, strconv.FormatFloat(value, 'g', -1, 64))
		}

		if err := writer.Write(record); err != nil {
			return fmt.Errorf("write %s: %w", path, err)
		}
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}

	return f.Close()
}
