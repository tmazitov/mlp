package vector

type numberType interface {
	~int | ~float64
}

type Vector[K numberType] []K

func (v Vector[K]) Copy() Vector[K] {

	result := make(Vector[K], 0, len(v))

	for _, value := range v {
		result = append(result, value)
	}

	return result
}

func (v Vector[K]) IsZero() bool {
	return len(v) == 0
}

func IsEqualDimension[K numberType](vectors ...Vector[K]) bool {

	if len(vectors) == 0 || len(vectors) == 1 {
		return true
	}

	init := vectors[0]

	for _, v := range vectors {
		if len(init) != len(v) {
			return false
		}
	}

	return true
}
