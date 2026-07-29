package analytics

import "errors"

var (
	ErrInvalidBatchSize = errors.New("batchReader error: argument batchSize can't be negative")
)
