package vector

func (v Vector[K]) Dot(u Vector[K]) K {

	var result K
	for i := range v {
		result = fma(v[i], u[i], result)
	}

	return result
}
