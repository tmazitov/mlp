package charts

import (
	"fmt"
	"path/filepath"

	"mlp/internal/analytics"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// ClassBalance charts the number of benign vs malignant rows. A skewed
// balance is a strong hint to use a weighted loss or resampling, or the
// network will learn to just predict the majority class.
//
// Values: ds.ClassCounts() just tallies how many rows have diagnosis "B"
// and how many have "M" — one bar per class, height = row count.
func ClassBalance(ds *analytics.Dataset, outDir string) error {
	counts := ds.ClassCounts()

	values := plotter.Values{float64(counts["B"]), float64(counts["M"])}
	bars, err := plotter.NewBarChart(values, vg.Points(80))
	if err != nil {
		return err
	}
	bars.Color = colorBenign

	p := plot.New()
	p.Title.Text = "Diagnosis class balance"
	p.Y.Label.Text = "rows"
	p.Add(bars)
	p.NominalX(
		fmt.Sprintf("Benign (%d)", counts["B"]),
		fmt.Sprintf("Malignant (%d)", counts["M"]),
	)

	return p.Save(5*vg.Inch, 4*vg.Inch, filepath.Join(outDir, "class_balance.png"))
}
