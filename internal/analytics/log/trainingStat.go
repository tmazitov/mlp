package log

type TrainingStat struct {
	Epoch       int
	AverageLoss float64
}

func NewTrainingStat(epoch int, loss []float64) TrainingStat {

	return TrainingStat{
		Epoch:       epoch,
		AverageLoss: averageLoss(loss),
	}
}

func averageLoss(loss []float64) float64 {

	if len(loss) == 0 {
		return 0
	}

	var sum float64
	for _, value := range loss {
		sum += value
	}

	return sum / float64(len(loss))
}
