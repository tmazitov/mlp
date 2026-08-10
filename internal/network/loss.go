package network

import (
	"math"
)

type lossFunc string

var (
	MSELossFunc          lossFunc = "mse"
	CrossEntropyLossFunc lossFunc = "crossEntropy"
)

func calculateLoss(loss lossFunc, answer, predict float64) float64 {
	switch loss {
	case CrossEntropyLossFunc:
		return crossEntropyLoss(answer, predict)
	}
	return 0
}

func crossEntropyLoss(answer, predict float64) float64 {
	return -(answer*math.Log(predict) + (1-answer)*math.Log(1-predict))
}
