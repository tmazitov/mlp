package charts

import (
	"fmt"
	"image/color"
	"os"
	"path/filepath"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

var (
	colorTrain   = color.RGBA{R: 130, G: 90, B: 200, A: 255}
	colorVal     = color.RGBA{R: 221, G: 132, B: 82, A: 255}
	colorMinLoss = color.RGBA{R: 160, G: 160, B: 160, A: 255}
)

// LossCurve charts training and validation loss for each epoch, so it's
// easy to see at a glance whether a run actually converged (both trending
// down) or overfit (training loss keeps dropping while validation loss
// flattens or climbs back up). A dashed gray line marks the minimum
// validation loss reached, with its exact value called out as an extra
// tick on the Y axis.
// trainLosses[i]/valLosses[i] are the average loss recorded for epoch i+1.
// It returns the path of the PNG it wrote.
func LossCurve(trainLosses, valLosses []float64, outDir string) (string, error) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", fmt.Errorf("create output dir: %w", err)
	}

	trainLine, err := newSeries(trainLosses, colorTrain)
	if err != nil {
		return "", err
	}
	valLine, err := newSeries(valLosses, colorVal)
	if err != nil {
		return "", err
	}

	minValLoss := minOf(valLosses)
	minLine, err := plotter.NewLine(plotter.XYs{
		{X: 1, Y: minValLoss},
		{X: float64(len(valLosses)), Y: minValLoss},
	})
	if err != nil {
		return "", err
	}
	minLine.Color = colorMinLoss
	minLine.Width = vg.Points(1)
	minLine.Dashes = []vg.Length{vg.Points(4), vg.Points(3)}

	p := plot.New()
	p.Title.Text = "Loss by epoch"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "average loss"
	p.Add(trainLine, valLine, minLine)

	p.Legend.Add("training loss", trainLine)
	p.Legend.Add("validation loss", valLine)
	p.Legend.Add(fmt.Sprintf("min val %.4f", minValLoss), minLine)
	p.Legend.Top = true

	// Keep the usual auto-generated ticks, and add one more exactly at the
	// minimum so its value is readable straight off the axis.
	p.Y.Tick.Marker = tickerWithHighlight(minValLoss)

	path := filepath.Join(outDir, "loss_curve.png")
	if err := p.Save(7*vg.Inch, 4*vg.Inch, path); err != nil {
		return "", err
	}
	return path, nil
}
