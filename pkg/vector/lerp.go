package vector

func Lerp[K numberType](v, u Vector[K], t float32) Vector[K] {

	if !IsEqualDimension(v, u) || t < 0 || t > 1 {
		return nil
	}

	result := make(Vector[K], len(v))

	for i := range v {
		result[i] = fma(u[i]-v[i], K(t), v[i])
	}
	return result
}
