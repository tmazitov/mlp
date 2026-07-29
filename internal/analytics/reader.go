package analytics

import "io"

type batchReader struct {
	state   int
	dataset *Dataset
}

func newBatchReader(dataset *Dataset) *batchReader {
	return &batchReader{
		dataset: dataset,
		state:   0,
	}
}

func (r *batchReader) Read(batchSize int) ([]Row, error) {
	if batchSize < 0 {
		return nil, ErrInvalidBatchSize
	}
	if r.state >= len(r.dataset.Rows) {
		return nil, io.EOF
	}

	result := r.dataset.Rows[r.state : r.state+batchSize]
	r.state += batchSize

	return result, nil
}

func (r *batchReader) Reset() {
	r.state = 0
}
