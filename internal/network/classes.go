package network

type predictedClass string

var (
	MalignantClass predictedClass = "M"
	BenignClass    predictedClass = "B"
)

// classFromIndex maps an output-vector position back to a class. The order
// is fixed by analytics.Row.DiagnosisVector, which encodes "M" as [1,0] and
// "B" as [0,1] — index 0 is malignant. Both ends of that convention have to
// agree, so prediction reads it from here rather than repeating it.
func classFromIndex(index int) predictedClass {
	if index == 0 {
		return MalignantClass
	}
	return BenignClass
}
