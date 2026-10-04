package charts

import (
	"fmt"
	"image/color"
	"os"
	"path/filepath"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/vg"
)

// LabeledSeries is one model's metric over the epochs, under the name it
// should carry in the legend.
type LabeledSeries struct {
	Name   string
	Values []float64
}

// comparisonPalette gives each model its own line colour. Models are
// compared by eye, so the colours have to stay apart from one another
// rather than form a gradient.
var comparisonPalette = []color.Color{
	color.RGBA{R: 130, G: 90, B: 200, A: 255},  // violet
	color.RGBA{R: 221, G: 132, B: 82, A: 255},  // orange
	color.RGBA{R: 76, G: 114, B: 176, A: 255},  // blue
	color.RGBA{R: 85, G: 168, B: 104, A: 255},  // green
	color.RGBA{R: 196, G: 78, B: 82, A: 255},   // red
	color.RGBA{R: 129, G: 114, B: 179, A: 255}, // purple
	color.RGBA{R: 147, G: 120, B: 96, A: 255},  // brown
	color.RGBA{R: 100, G: 181, B: 205, A: 255}, // cyan
}

// ComparisonCurve draws one line per model on shared axes, which is the
// point of running several configurations: the gap between the lines is
// the answer, and it is only readable when they share a scale.
//
// These are validation curves. Plotting each model's training curve too
// would double the lines for information that says how well a model fits
// rows it has already seen, which is not what is being compared.
func ComparisonCurve(series []LabeledSeries, title, yLabel, fileName, outDir string) (string, error) {
	if len(series) == 0 {
		return "", fmt.Errorf("%s: nothing to compare", fileName)
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", fmt.Errorf("create output dir: %w", err)
	}

	p := plot.New()
	p.Title.Text = title
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = yLabel

	for i, s := range series {
		if len(s.Values) == 0 {
			continue
		}

		line, err := newSeries(s.Values, comparisonPalette[i%len(comparisonPalette)])
		if err != nil {
			return "", err
		}

		p.Add(line)
		p.Legend.Add(s.Name, line)
	}

	p.Legend.Top = true

	path := filepath.Join(outDir, fileName)
	if err := p.Save(8*vg.Inch, 5*vg.Inch, path); err != nil {
		return "", err
	}
	return path, nil
}
