package neuron

import (
	"math"

	"mlp/pkg/vector"
)

type OptimizerFunc string

var (
	SGDOptimizer      OptimizerFunc = "sgd"
	NesterovOptimizer OptimizerFunc = "nesterov"
	RMSPropOptimizer  OptimizerFunc = "rmsprop"
)

// Defaults for the knobs the optimizers carry beyond the learning rate.
// These are the values the original papers and every mainstream framework
// settle on; exposing them in the UI would add three fields nobody tunes.
const (
	// defaultMomentum is how much of the previous step is carried into the
	// next one.
	defaultMomentum = 0.9

	// defaultDecay is how fast the running average of squared gradients
	// forgets older batches.
	defaultDecay = 0.9

	// defaultEpsilon keeps the RMSProp division finite when a parameter's
	// gradient has been zero for a while.
	defaultEpsilon = 1e-8
)

// optimizer turns an averaged gradient into the step actually subtracted
// from a parameter. Each neuron owns its own instance, because momentum and
// squared-gradient averages are per-parameter state carried between
// updates — sharing one across neurons would blur their histories together.
type optimizer interface {
	// step returns the amounts to subtract from the weights and the bias.
	step(gradWeights vector.Vector[float64], gradBias float64) (vector.Vector[float64], float64)
}

func newOptimizer(kind OptimizerFunc, learningRate float64) optimizer {
	switch kind {
	case NesterovOptimizer:
		return &nesterov{learningRate: learningRate, momentum: defaultMomentum}
	case RMSPropOptimizer:
		return &rmsProp{learningRate: learningRate, decay: defaultDecay, epsilon: defaultEpsilon}
	default:
		return &sgd{learningRate: learningRate}
	}
}

// sgd takes the gradient at face value: step = lr * g.
type sgd struct {
	learningRate float64
}

func (o *sgd) step(gradWeights vector.Vector[float64], gradBias float64) (vector.Vector[float64], float64) {
	return gradWeights.Scl(o.learningRate), gradBias * o.learningRate
}

// nesterov accelerates along directions the gradient keeps pointing in,
// while damping the zig-zagging that plain SGD does across a ravine.
//
// This is the reformulation from Sutskever et al. (2013), the one every
// framework implements:
//
//	v = momentum*v + g
//	step = lr * (g + momentum*v)
//
// The textbook version evaluates the gradient at a look-ahead position,
// momentum steps away from the current weights. That would mean shifting
// every weight, running a forward pass and shifting back — the
// reformulation is algebraically the same update using the gradient we
// already have.
type nesterov struct {
	learningRate float64
	momentum     float64

	velocityWeights vector.Vector[float64]
	velocityBias    float64
}

func (o *nesterov) step(gradWeights vector.Vector[float64], gradBias float64) (vector.Vector[float64], float64) {
	if o.velocityWeights == nil {
		o.velocityWeights = make(vector.Vector[float64], len(gradWeights))
	}

	stepWeights := make(vector.Vector[float64], len(gradWeights))
	for i, gradient := range gradWeights {
		o.velocityWeights[i] = o.momentum*o.velocityWeights[i] + gradient
		stepWeights[i] = o.learningRate * (gradient + o.momentum*o.velocityWeights[i])
	}

	o.velocityBias = o.momentum*o.velocityBias + gradBias
	stepBias := o.learningRate * (gradBias + o.momentum*o.velocityBias)

	return stepWeights, stepBias
}

// rmsProp divides each parameter's step by the root of a running average of
// its recent squared gradients:
//
//	s = decay*s + (1-decay)*g²
//	step = lr * g / (sqrt(s) + epsilon)
//
// Parameters whose gradients are consistently large get smaller steps and
// parameters starved of signal get larger ones, so one learning rate suits
// the whole network instead of being a compromise across it.
//
// Because the step is normalised by gradient magnitude rather than scaled
// by it, the useful learning rate is far smaller than for plain SGD —
// around 0.001 rather than 0.05.
type rmsProp struct {
	learningRate float64
	decay        float64
	epsilon      float64

	squaredWeights vector.Vector[float64]
	squaredBias    float64
}

func (o *rmsProp) step(gradWeights vector.Vector[float64], gradBias float64) (vector.Vector[float64], float64) {
	if o.squaredWeights == nil {
		o.squaredWeights = make(vector.Vector[float64], len(gradWeights))
	}

	stepWeights := make(vector.Vector[float64], len(gradWeights))
	for i, gradient := range gradWeights {
		o.squaredWeights[i] = o.decay*o.squaredWeights[i] + (1-o.decay)*gradient*gradient
		stepWeights[i] = o.learningRate * gradient / (math.Sqrt(o.squaredWeights[i]) + o.epsilon)
	}

	o.squaredBias = o.decay*o.squaredBias + (1-o.decay)*gradBias*gradBias
	stepBias := o.learningRate * gradBias / (math.Sqrt(o.squaredBias) + o.epsilon)

	return stepWeights, stepBias
}
