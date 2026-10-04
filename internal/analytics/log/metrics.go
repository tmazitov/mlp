package log

// Metrics scores a set of predictions against their true classes.
//
// Malignant is the positive class: in this dataset a false negative is a
// malignant tumour reported as benign, which is a far more costly mistake
// than the reverse. Accuracy alone averages the two together and hides
// that asymmetry, which is why the counts are kept and recall is reported
// beside it.
type Metrics struct {
	Rows int
	Loss float64

	TruePositives  int
	TrueNegatives  int
	FalsePositives int
	FalseNegatives int
}

// Accuracy is the share of rows classified correctly.
func (m Metrics) Accuracy() float64 {
	return ratio(m.TruePositives+m.TrueNegatives, m.Rows)
}

// Precision: of the rows called malignant, how many were. Low precision
// means false alarms.
func (m Metrics) Precision() float64 {
	return ratio(m.TruePositives, m.TruePositives+m.FalsePositives)
}

// Recall: of the malignant rows, how many were caught. This is the one
// that matters most here — what it misses is a missed cancer.
func (m Metrics) Recall() float64 {
	return ratio(m.TruePositives, m.TruePositives+m.FalseNegatives)
}

// Specificity: of the benign rows, how many were left alone.
func (m Metrics) Specificity() float64 {
	return ratio(m.TrueNegatives, m.TrueNegatives+m.FalsePositives)
}

// F1 is the harmonic mean of precision and recall, which only rises when
// both do — unlike their average, which a model can lift by sacrificing
// one of them.
func (m Metrics) F1() float64 {
	precision, recall := m.Precision(), m.Recall()
	if precision+recall == 0 {
		return 0
	}
	return 2 * precision * recall / (precision + recall)
}

// Add records one prediction against its true class.
func (m *Metrics) Add(predictedMalignant, actuallyMalignant bool) {
	m.Rows++
	switch {
	case predictedMalignant && actuallyMalignant:
		m.TruePositives++
	case !predictedMalignant && !actuallyMalignant:
		m.TrueNegatives++
	case predictedMalignant && !actuallyMalignant:
		m.FalsePositives++
	default:
		m.FalseNegatives++
	}
}

// ratio guards the divisions above: a model that never predicts malignant
// has no precision to speak of, and reporting NaN would poison every chart
// and average downstream.
func ratio(part, whole int) float64 {
	if whole == 0 {
		return 0
	}
	return float64(part) / float64(whole)
}
