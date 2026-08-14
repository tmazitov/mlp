package network

import (
	"mlp/pkg/vector"
)

type layerCache struct {
	activation vector.Vector[float64]
	derivative vector.Vector[float64]
	output     vector.Vector[float64]
}

func newLayerCache(size int) layerCache {
	return layerCache{
		activation: make(vector.Vector[float64], size),
		derivative: make(vector.Vector[float64], size),
		output:     make(vector.Vector[float64], size),
	}
}

func (f *layerCache) clean() {
	clear(f.activation)
	clear(f.derivative)
	clear(f.output)
}
