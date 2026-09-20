package charts

import (
	"fmt"
	"image/color"
	"math"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// newSeries turns values (one per epoch, 1-indexed on the X axis) into a
// styled line — shared by the loss and accuracy curves' training/
// validation lines.
func newSeries(values []float64, col color.Color) (*plotter.Line, error) {
	points := make(plotter.XYs, len(values))
	for i, v := range values {
		points[i] = plotter.XY{X: float64(i + 1), Y: v}
	}

	line, err := plotter.NewLine(points)
	if err != nil {
		return nil, err
	}
	line.Color = col
	line.Width = vg.Points(1.5)
	return line, nil
}

func minOf(values []float64) float64 {
	min := values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
	}
	return min
}

func maxOf(values []float64) float64 {
	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}

// tickerWithHighlight behaves like plot.DefaultTicks, but adds one extra
// tick at value — unless an auto-generated tick already lands close enough
// that the two labels would overlap, in which case the default ticks are
// left alone.
func tickerWithHighlight(value float64) plot.Ticker {
	defaultTicks := plot.DefaultTicks{}
	return plot.TickerFunc(func(min, max float64) []plot.Tick {
		ticks := defaultTicks.Ticks(min, max)

		threshold := (max - min) * 0.05
		for _, t := range ticks {
			if !t.IsMinor() && math.Abs(t.Value-value) < threshold {
				return ticks
			}
		}

		return append(ticks, plot.Tick{Value: value, Label: fmt.Sprintf("%.4f", value)})
	})
}
