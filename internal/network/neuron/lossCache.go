package neuron

import "mlp/pkg/vector"

type lossCache struct {
	weights []vector.Vector[float64]
	bias    []float64
}

func (c *lossCache) AddRecord(weightsLoss vector.Vector[float64], biasLoss float64) {
	c.weights = append(c.weights, weightsLoss)
	c.bias = append(c.bias, biasLoss)
}

func (c *lossCache) GetAverage() (vector.Vector[float64], float64) {

	if len(c.bias) == 0 || len(c.weights) == 0 || len(c.weights) != len(c.bias) {
		return vector.Vector[float64]{}, 0
	}

	var (
		avgWeighs = make(vector.Vector[float64], len(c.weights[0]))
		avgBias   = float64(0)
	)

	for i, weight := range c.weights {
		avgBias += c.bias[i]

		for j, weightElem := range weight {
			avgWeighs[j] += weightElem
		}
	}

	avgBias = avgBias / float64(len(c.bias))
	avgWeighs = avgWeighs.Scl(1 / float64(len(c.bias)))

	return avgWeighs, avgBias
}

func (c *lossCache) Clear() {
	c.weights = []vector.Vector[float64]{}
	c.bias = []float64{}
}
