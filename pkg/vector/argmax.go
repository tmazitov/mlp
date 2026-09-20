package vector

// ArgMax returns the index of the largest element in v, or -1 if v is
// empty. Used to turn a softmax probability vector into a predicted class
// index.
func (v Vector[K]) ArgMax() int {
	if len(v) == 0 {
		return -1
	}

	best := 0
	for i, val := range v {
		if val > v[best] {
			best = i
		}
	}
	return best
}
