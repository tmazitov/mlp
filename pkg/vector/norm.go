package vector

func (v Vector[K]) Norm1() float32 {

	var sum float32 = 0.0

	if len(v) == 0 {
		return 0
	}

	for _, value := range v {
		if value < 0 {
			value *= -1
		}
		sum += float32(value)
	}

	return sum
}

// sqrt approximates the square root via Newton's method, using only
// elementary arithmetic (this exercise's allowed math functions list
// is limited to fused multiply-add, so math.Sqrt/math.Pow are off-limits).
func sqrt(x float64) float64 {

	if x <= 0 {
		return 0
	}

	guess := x
	for range 40 {
		guess = fma(0.5, guess, 0.5*x/guess)
	}

	return guess
}

func (v Vector[K]) Norm() float32 {

	var sum float64 = 0.0

	if len(v) == 0 {
		return 0
	}

	for _, value := range v {
		f := float64(value)
		sum = fma(f, f, sum)
	}

	return float32(sqrt(sum))
}

func (v Vector[K]) NormInf() float32 {

	if len(v) == 0 {
		return 0
	}

	max := float64(v[0])
	if max < 0 {
		max *= -1
	}

	for _, value := range v {

		if value < 0 {
			value *= -1
		}

		if float64(value) > max {
			max = float64(value)
		}
	}
	return float32(max)
}
