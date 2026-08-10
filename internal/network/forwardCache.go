package network

import (
	"mlp/pkg/vector"
)

type forwardCache struct {
	activation vector.Vector[float64]
	output     vector.Vector[float64]
}

func newForwardCache(size int) forwardCache {
	return forwardCache{
		activation: make(vector.Vector[float64], size),
		output:     make(vector.Vector[float64], size),
	}
}

func (f *forwardCache) clean() {
	clear(f.activation)
	clear(f.output)
}
