package network

import "mlp/pkg/vector"

// snapshot is a copy of every weight and bias in the network.
//
// Early stopping needs this because the best model is rarely the last one:
// once validation loss turns back up, the weights that produced the lowest
// one have already been overwritten by the epochs that followed. Keeping a
// copy is what makes "stop and go back" possible rather than just "stop".
type snapshot [][]neuronParameters

type neuronParameters struct {
	weights vector.Vector[float64]
	bias    float64
}

// takeSnapshot copies the current parameters. The weights are copied
// rather than aliased: the originals keep being written to by the very
// next batch.
func (m MLP) takeSnapshot() snapshot {
	snap := make(snapshot, len(m.layers))

	for i, layer := range m.layers {
		snap[i] = make([]neuronParameters, len(layer.neurons))
		for j, n := range layer.neurons {
			weights := make(vector.Vector[float64], len(n.Weight()))
			copy(weights, n.Weight())
			snap[i][j] = neuronParameters{weights: weights, bias: n.Bias()}
		}
	}

	return snap
}

// restore puts a snapshot back into the network.
func (m MLP) restore(snap snapshot) {
	for i, layer := range m.layers {
		if i >= len(snap) {
			return
		}
		for j, n := range layer.neurons {
			if j >= len(snap[i]) {
				break
			}
			n.SetParameters(snap[i][j].weights, snap[i][j].bias)
		}
	}
}
