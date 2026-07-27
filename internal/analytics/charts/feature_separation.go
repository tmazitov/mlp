package charts

import (
	"math/rand/v2"
	"path/filepath"
	"strings"

	"mlp/internal/analytics"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// FeatureSeparation charts standardized benign-vs-malignant values for
// every "_mean" feature as a jittered strip plot — every row is drawn as
// its own point rather than collapsed into a box-plot summary. A feature
// where the blue (benign) and orange (malignant) point clouds barely
// overlap carries a lot of discriminating signal; one where they're mixed
// together carries little.
//
// Values: for each "_mean" feature, ds.ValuesByDiagnosis(idx) splits its
// raw per-row values by diagnosis. Each value is standardized to a
// z-score, (x-mean)/stddev, using that feature's own overall mean/stddev
// from ds.FeatureStats() — the same transform you'd apply before feeding
// this data into the MLP, and what makes it valid to put all 10 features
// on one shared y-axis (their raw scales differ by orders of magnitude,
// see FeatureScales). Each point is nudged sideways by a small
// deterministically-seeded random offset (jitter) purely so overlapping
// points don't stack directly on top of one another — it has no effect on
// the underlying value.
func FeatureSeparation(ds *analytics.Dataset, outDir string) error {
	var featureIdx []int
	for i, name := range analytics.FeatureNames {
		if strings.HasSuffix(name, "_mean") {
			featureIdx = append(featureIdx, i)
		}
	}

	stats := ds.FeatureStats()
	rng := rand.New(rand.NewPCG(1, 1))

	const (
		groupOffset = 0.22 // horizontal gap between the benign/malignant clouds
		jitterWidth = 0.12 // +/- random spread within each cloud
	)

	var benignPts, malignantPts plotter.XYs
	labels := make([]string, len(featureIdx))

	for x, idx := range featureIdx {
		values := ds.ValuesByDiagnosis(idx)
		mean, stddev := stats[idx].Mean, stats[idx].StdDev

		for _, v := range values["B"] {
			benignPts = append(benignPts, plotter.XY{
				X: float64(x) - groupOffset + jitter(rng, jitterWidth),
				Y: (v - mean) / stddev,
			})
		}
		for _, v := range values["M"] {
			malignantPts = append(malignantPts, plotter.XY{
				X: float64(x) + groupOffset + jitter(rng, jitterWidth),
				Y: (v - mean) / stddev,
			})
		}

		labels[x] = strings.TrimSuffix(analytics.FeatureNames[idx], "_mean")
	}

	benignScatter, err := plotter.NewScatter(benignPts)
	if err != nil {
		return err
	}
	benignScatter.Color = colorBenign
	benignScatter.Radius = vg.Points(1.5)

	malignantScatter, err := plotter.NewScatter(malignantPts)
	if err != nil {
		return err
	}
	malignantScatter.Color = colorMalignant
	malignantScatter.Radius = vg.Points(1.5)

	p := plot.New()
	p.Title.Text = "Feature separation, standardized (blue=Benign, orange=Malignant)"
	p.Y.Label.Text = "standardized value (z-score)"
	p.Add(benignScatter, malignantScatter)

	p.NominalX(labels...)
	p.X.Tick.Label.Rotation = 1.5708 // pi/2, so the 10 feature names don't overlap
	p.X.Tick.Label.XAlign = draw.XRight
	p.X.Tick.Label.YAlign = draw.YCenter

	return p.Save(11*vg.Inch, 6*vg.Inch, filepath.Join(outDir, "feature_separation.png"))
}

func jitter(rng *rand.Rand, width float64) float64 {
	return (rng.Float64()*2 - 1) * width
}
