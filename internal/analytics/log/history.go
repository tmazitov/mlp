package log

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

// historyColumns is both the header and the order WriteHistory writes, so
// the two cannot drift apart.
var historyColumns = []string{
	"epoch",
	"train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1", "train_specificity",
	"val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1", "val_specificity",
	"val_tp", "val_tn", "val_fp", "val_fn",
}

// WriteHistory records every epoch of a run as CSV, named after the run's
// start time so successive runs accumulate rather than overwrite each
// other. It returns the path it wrote.
//
// The charts show the shape of a run; this keeps the numbers behind them,
// which is what you need to compare a run against one from last week, or
// to look up where exactly a curve turned.
func WriteHistory(stats []TrainingStat, label, outDir string) (string, error) {
	if len(stats) == 0 {
		return "", fmt.Errorf("history: nothing to write")
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", fmt.Errorf("create output dir: %w", err)
	}

	name := fmt.Sprintf("history_%s.csv", time.Now().Format("20060102_150405"))
	if label != "" {
		name = fmt.Sprintf("history_%s_%s.csv", sanitise(label), time.Now().Format("20060102_150405"))
	}
	path := filepath.Join(outDir, name)

	f, err := os.Create(path)
	if err != nil {
		return "", fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	if err := writer.Write(historyColumns); err != nil {
		return "", fmt.Errorf("write %s: %w", path, err)
	}

	for _, stat := range stats {
		record := []string{
			strconv.Itoa(stat.Epoch + 1),
			number(stat.Train.Loss), number(stat.Train.Accuracy()), number(stat.Train.Precision()),
			number(stat.Train.Recall()), number(stat.Train.F1()), number(stat.Train.Specificity()),
			number(stat.Val.Loss), number(stat.Val.Accuracy()), number(stat.Val.Precision()),
			number(stat.Val.Recall()), number(stat.Val.F1()), number(stat.Val.Specificity()),
			strconv.Itoa(stat.Val.TruePositives), strconv.Itoa(stat.Val.TrueNegatives),
			strconv.Itoa(stat.Val.FalsePositives), strconv.Itoa(stat.Val.FalseNegatives),
		}
		if err := writer.Write(record); err != nil {
			return "", fmt.Errorf("write %s: %w", path, err)
		}
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		return "", fmt.Errorf("write %s: %w", path, err)
	}

	return path, f.Close()
}

func number(value float64) string {
	return strconv.FormatFloat(value, 'f', 6, 64)
}

// sanitise keeps a model's name usable as part of a filename.
func sanitise(label string) string {
	cleaned := make([]rune, 0, len(label))
	for _, r := range label {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			cleaned = append(cleaned, r)
		case r == '.' || r == '-' || r == '_':
			cleaned = append(cleaned, r)
		default:
			cleaned = append(cleaned, '_')
		}
	}
	return string(cleaned)
}
