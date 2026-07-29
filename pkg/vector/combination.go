package vector

import (
	"math"
)

func fma[K numberType](x, y, z K) K {
	return K(math.FMA(float64(x), float64(y), float64(z)))
}

func LinearCombination[K numberType](u []Vector[K], coefs []K) Vector[K] {

	if len(u) == 0 || len(u) != len(coefs) || !IsEqualDimension(u...) {
		return nil
	}

	result := make(Vector[K], len(u[0]))
	for i, vector := range u {
		for j, value := range vector {
			result[j] = fma(value, coefs[i], result[j])
		}
	}

	return result
}
