package neuron

import (
	"math"
	"math/rand/v2"
	"mlp/pkg/vector"
)

// https://vc.ru/id4616024/2264108-inizializatsiya-vesov-v-nevrosnetyakh-metody-xavier-i-he

// 1. Why we can't put all weight to be 0? Or 1? Or another const value?
//
// The thing is if you allow that, results of activation functions from different neurons
// will respond this the same value. That means, whole layer of N neurons will collapse to 1 neuron,
// that can't learn features any longer.
//
// In terms of Machine Learning, this situation calls |Symmetry Problem|

// 2. What will be if I put large values?
//
// During whole training phase, model transfer values from layer to layer.
// Every layer multiply input data by its weight. If the weights are large enough,
// activation vector will be large too. Next layer receive large activation vector,
// and in that way 1-2 layers the model stops training. Sigmoid and Gradient Decent just go out of their ranges.
//
// In terms of Machine Learning, this situation calls |Exploding Gradients|

// 3. Why we can't put too small values?
//
// In the same way as previous paragraph, layer by layer model activation vectors
// will decrease and lose the features signal.
//
// In terms of Machine Learning, this situation calls |Vanishing Gradients|

type WeightInitFunc string

var (
	XavierInitFunc WeightInitFunc = "xavier"
	HeInitFunc     WeightInitFunc = "he"
)

// Let Var be function of dispersion, n count of input values.
// Then dispersion of layer will be:
//
//  Var(y) = Var(w) × Var(x) × n
//
// To keep network stable, it should stick to this condition Var(y) ≈ Var(x)
// In terms of Machine Learning, this condition calls |Signal Balance Maintaining (SBM)|
//
// Let x = x₁, x₂, …, xₙ be input vector and w = w₁, w₂, …, wₙ weight vector of specific neuron.
// Then output will be:
//
//  y = Σ(wᵢ·xᵢ).
//
// Let x be a vector of independent values with average equal to 0. Then:
//
// Var(y) = n × Var(w) × Var(x).
//
// To keep SBM (above) condition be true,
//
// Var(w) = 1 / n ---> basic principle of all algorithms

func xavierInitVector(nIn, nOut uint) vector.Vector[float64] {

	result := make(vector.Vector[float64], nIn)

	for i := range result {
		result[i] = xavierInit(nIn, nOut)
	}

	return result
}

func xavierInit(nIn, nOut uint) float64 {
	limit := math.Sqrt(6.0 / float64(nIn+nOut))
	return rand.Float64()*(2*limit) - limit
}

func heInit(nIn uint) float64 {
	std := math.Sqrt(2.0 / float64(nIn))
	return rand.NormFloat64() * std
}

func heInitVector(nIn uint) vector.Vector[float64] {
	result := make(vector.Vector[float64], nIn)

	for i := range result {
		result[i] = heInit(nIn)
	}

	return result
}
