package log

// TrainingStat reports one epoch's metrics on both sets. The training
// numbers say how well the network fits rows it is learning from; the
// validation ones say how well that generalises, and are the ones worth
// judging a run by.
type TrainingStat struct {
	Epoch int
	Train Metrics
	Val   Metrics

	// EarlyStopped marks the final stat of a run that stopped short, with
	// BestEpoch naming the epoch whose weights were restored. The metrics
	// on this stat are still the ones just measured, not the best ones —
	// they are what triggered the stop.
	EarlyStopped bool
	BestEpoch    int
}
