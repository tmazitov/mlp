package analytics

import "math/rand/v2"

// Split randomly partitions d's rows into two new datasets: roughly
// trainRatio of the rows for training, the rest for validation. Rows are
// shuffled first so a dataset ordered by class (or anything else) doesn't
// produce a lopsided split. d itself is left untouched.
func (d *Dataset) Split(trainRatio float64) (train, val *Dataset) {
	shuffled := make([]Row, len(d.Rows))
	copy(shuffled, d.Rows)
	rand.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	cut := int(float64(len(shuffled)) * trainRatio)

	trainRows := make([]Row, cut)
	copy(trainRows, shuffled[:cut])

	valRows := make([]Row, len(shuffled)-cut)
	copy(valRows, shuffled[cut:])

	return &Dataset{Rows: trainRows}, &Dataset{Rows: valRows}
}
