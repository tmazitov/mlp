package network

import (
	"math"

	"mlp/pkg/vector"
)

type lossFunc string

// Cross-entropy is the only loss implemented, and it is what the backward
// pass optimizes through the softmax+cross-entropy shortcut in Train
// (predict - answer). Reporting any other metric would describe a different
// model than the one actually being trained, so Train rejects anything else
// rather than quietly substituting it.
var (
	CrossEntropyLossFunc lossFunc = "crossEntropy"
)

// logEpsilon keeps ln(p) finite. Softmax can underflow to exactly 0 for a
// class the model is confidently wrong about, and math.Log(0) is -Inf — one
// such row would poison the whole epoch average and every chart drawn from
// it.
const logEpsilon = 1e-15

// crossEntropyLoss returns the categorical cross-entropy, -Σ yᵢ·ln(pᵢ).
// With a one-hot answer every term where yᵢ=0 drops out, so it collapses to
// -ln(p) of the true class. For two softmax outputs (p₀+p₁=1) that equals
// the binary form in the subject: -[y·ln(p) + (1-y)·ln(1-p)].
func crossEntropyLoss(answer, predict vector.Vector[float64]) float64 {
	var sum float64

	for i, y := range answer {
		if y == 0 {
			continue
		}
		sum -= y * math.Log(math.Max(predict[i], logEpsilon))
	}

	return sum
}
