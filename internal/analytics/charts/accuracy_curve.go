package charts

import (
	"fmt"
	"os"
	"path/filepath"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// AccuracyCurve charts training and validation classification accuracy for
// each epoch. Diverging curves (training accuracy climbing while
// validation accuracy stalls or drops) is the classic sign of overfitting.
// A dashed gray line marks the best (highest) validation accuracy reached.
// trainAcc[i]/valAcc[i] are the accuracy recorded for epoch i+1. It
// returns the path of the PNG it wrote.
func AccuracyCurve(trainAcc, valAcc []float64, outDir string) (string, error) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", fmt.Errorf("create output dir: %w", err)
	}

	trainLine, err := newSeries(trainAcc, colorTrain)
	if err != nil {
		return "", err
	}
	valLine, err := newSeries(valAcc, colorVal)
	if err != nil {
		return "", err
	}

	bestValAcc := maxOf(valAcc)
	bestLine, err := plotter.NewLine(plotter.XYs{
		{X: 1, Y: bestValAcc},
		{X: float64(len(valAcc)), Y: bestValAcc},
	})
	if err != nil {
		return "", err
	}
	bestLine.Color = colorMinLoss
	bestLine.Width = vg.Points(1)
	bestLine.Dashes = []vg.Length{vg.Points(4), vg.Points(3)}

	p := plot.New()
	p.Title.Text = "Accuracy by epoch"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "accuracy"
	p.Add(trainLine, valLine, bestLine)

	p.Legend.Add("training acc", trainLine)
	p.Legend.Add("validation acc", valLine)
	p.Legend.Add(fmt.Sprintf("best val %.4f", bestValAcc), bestLine)
	// Accuracy climbs toward the top as epochs progress (the mirror image
	// of the loss curve), so the legend goes in the bottom-right — the one
	// corner that stays empty — instead of overlapping the lines up top.

	p.Y.Tick.Marker = tickerWithHighlight(bestValAcc)

	path := filepath.Join(outDir, "accuracy_curve.png")
	if err := p.Save(7*vg.Inch, 4*vg.Inch, path); err != nil {
		return "", err
	}
	return path, nil
}
