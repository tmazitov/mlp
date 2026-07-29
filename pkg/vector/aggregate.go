package vector

func (v Vector[K]) Add(w Vector[K]) Vector[K] {

	if !IsEqualDimension(w, v) {
		return nil
	}

	values := make([]K, 0, len(v))
	for i, elem := range v {
		values = append(values, elem+w[i])
	}

	return (Vector[K])(values)
}

func (v Vector[K]) Sub(w Vector[K]) Vector[K] {

	if !IsEqualDimension(w, v) {
		return nil
	}

	values := make([]K, 0, len(v))
	for i, elem := range v {
		values = append(values, elem-w[i])
	}

	return (Vector[K])(values)
}

func (v Vector[K]) Scl(a K) Vector[K] {

	values := make([]K, 0, len(v))
	for _, elem := range v {
		values = append(values, elem*a)
	}

	return (Vector[K])(values)
}
