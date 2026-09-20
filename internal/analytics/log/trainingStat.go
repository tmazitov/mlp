package log

// TrainingStat reports one epoch's training and validation metrics: loss
// (average NormInf of predict-answer over the epoch's rows) and
// classification accuracy (share of rows where the predicted class,
// argmax(predict), matches the true one, argmax(answer)).
type TrainingStat struct {
	Epoch         int
	TrainLoss     float64
	ValLoss       float64
	TrainAccuracy float64
	ValAccuracy   float64
}
