package dataset

type Record struct {
	id        int
	diagnosis string
	features  []float64
}

func NewRecord(id int, diagnosis string, features []float64) *Record {

	return &Record{
		id:        id,
		diagnosis: diagnosis,
		features:  features,
	}
}
