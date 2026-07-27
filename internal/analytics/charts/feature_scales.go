package charts

import (
	"math"
	"path/filepath"
	"sort"

	"mlp/internal/analytics"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// FeatureScales charts log10(mean) for every feature. The 30 raw features
// span several orders of magnitude (area_mean ~650 vs smoothness_mean
// ~0.1) — feeding that straight into an MLP will make gradient descent
// dominated by the large-scale features, so this is the case for
// standardizing/normalizing inputs before training.
//
// Values: ds.FeatureStats() computes the plain mean of each of the 30
// features across every row (regardless of diagnosis). Each mean is then
// log10'd so the bars are comparable on one linear axis instead of one
// giant bar and 29 invisible slivers; features are sorted by mean,
// descending, so the spread is easy to read top to bottom.
func FeatureScales(ds *analytics.Dataset, outDir string) error {
	stats := ds.FeatureStats()
	sort.Slice(stats, func(i, j int) bool { return stats[i].Mean > stats[j].Mean })

	values := make(plotter.Values, len(stats))
	labels := make([]string, len(stats))
	for i, s := range stats {
		values[i] = math.Log10(s.Mean)
		labels[i] = abbreviateFeature(s.Name)
	}

	bars, err := plotter.NewBarChart(values, vg.Points(10))
	if err != nil {
		return err
	}
	bars.Horizontal = true
	bars.Color = colorMalignant

	p := plot.New()
	p.Title.Text = "Feature scale comparison (log10 of mean value)"
	p.X.Label.Text = "log10(mean value)"
	p.Y.Tick.Label.Font.Size = vg.Points(7)
	p.Add(bars)
	p.NominalY(labels...)

	return p.Save(7*vg.Inch, 10*vg.Inch, filepath.Join(outDir, "feature_scales.png"))
}
