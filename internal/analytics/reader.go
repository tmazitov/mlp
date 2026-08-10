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

	end := r.state + batchSize
	if end > len(r.dataset.Rows) {
		end = len(r.dataset.Rows)
	}

	result := r.dataset.Rows[r.state:end]
	r.state = end

	return result, nil
}

func (r *batchReader) Reset() {
	r.state = 0
}
