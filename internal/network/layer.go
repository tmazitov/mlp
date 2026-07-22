package network

type Layer struct {
	neurons []neuron
}

func NewLayer() *Layer {
	return &Layer{}
}
