// Package charts renders PNG charts from an analytics.Dataset that are
// meant to inform decisions before training the MLP: whether the classes
// are balanced, whether inputs need standardizing, which features are
// redundant, and which features actually separate the two classes.
package charts

import (
	"fmt"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"mlp/internal/analytics"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

var (
	colorBenign    = color.RGBA{R: 76, G: 114, B: 176, A: 255}
	colorMalignant = color.RGBA{R: 221, G: 132, B: 82, A: 255}
)

// GenerateAll renders every chart into outDir, creating it if needed.
func GenerateAll(ds *analytics.Dataset, outDir string) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	charts := []struct {
		name string
		fn   func(*analytics.Dataset, string) error
	}{
		{"class balance", ClassBalance},
		{"feature scales", FeatureScales},
		{"correlation heatmap", CorrelationHeatmap},
		{"feature separation", FeatureSeparation},
	}

	for _, c := range charts {
		if err := c.fn(ds, outDir); err != nil {
			return fmt.Errorf("%s: %w", c.name, err)
		}
	}

	return nil
}

// ClassBalance charts the number of benign vs malignant rows. A skewed
// balance is a strong hint to use a weighted loss or resampling, or the
// network will learn to just predict the majority class.
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

// FeatureScales charts log10(mean) for every feature. The 30 raw features
// span several orders of magnitude (area_mean ~650 vs smoothness_mean
// ~0.1) — feeding that straight into an MLP will make gradient descent
// dominated by the large-scale features, so this is the case for
// standardizing/normalizing inputs before training.
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

// CorrelationHeatmap charts the full 30x30 Pearson correlation matrix.
// Clusters of strong correlation (e.g. radius/perimeter/area, or a
// *_mean and its *_worst counterpart) point at redundant inputs worth
// dropping or compressing before training.
func CorrelationHeatmap(ds *analytics.Dataset, outDir string) error {
	grid := correlationGrid{matrix: ds.CorrelationMatrix()}

	pal := moreland.SmoothBlueRed()
	pal.SetMin(-1)
	pal.SetMax(1)

	heatmap := plotter.NewHeatMap(grid, pal.Palette(256))
	heatmap.Min, heatmap.Max = -1, 1

	ticks := make([]plot.Tick, len(analytics.FeatureNames))
	for i, name := range analytics.FeatureNames {
		ticks[i] = plot.Tick{Value: float64(i), Label: abbreviateFeature(name)}
	}

	p := plot.New()
	p.Title.Text = "Feature correlation heatmap (Pearson r)"
	p.Add(heatmap)

	p.X.Tick.Marker = plot.ConstantTicks(ticks)
	p.X.Tick.Label.Font.Size = vg.Points(6)
	p.X.Tick.Label.Rotation = 1.5708 // pi/2, so 30 labels don't overlap
	p.X.Tick.Label.XAlign = draw.XRight
	p.X.Tick.Label.YAlign = draw.YCenter

	p.Y.Tick.Marker = plot.ConstantTicks(ticks)
	p.Y.Tick.Label.Font.Size = vg.Points(6)

	return p.Save(13*vg.Inch, 12*vg.Inch, filepath.Join(outDir, "correlation_heatmap.png"))
}

// FeatureSeparation charts benign-vs-malignant box plots for every "_mean"
// feature, side by side. Features where the two boxes barely overlap (e.g.
// concave_points_mean) carry a lot of the discriminating signal; features
// where they mostly overlap (e.g. smoothness_mean) carry little.
func FeatureSeparation(ds *analytics.Dataset, outDir string) error {
	var featureIdx []int
	for i, name := range analytics.FeatureNames {
		if strings.HasSuffix(name, "_mean") {
			featureIdx = append(featureIdx, i)
		}
	}

	const cols = 5
	rows := (len(featureIdx) + cols - 1) / cols

	plots := make([][]*plot.Plot, rows)
	for r := range plots {
		plots[r] = make([]*plot.Plot, cols)
	}

	boxWidth := vg.Points(18)

	for k, idx := range featureIdx {
		values := ds.ValuesByDiagnosis(idx)

		// Both boxes sit at the same nominal x=0 category, offset left/right
		// of center — this is gonum's documented pattern for a grouped box
		// plot; putting them at two different x categories instead misplaces
		// the axes when the plots are tiled into a grid.
		benign, err := plotter.NewBoxPlot(boxWidth, 0, plotter.Values(values["B"]))
		if err != nil {
			return err
		}
		benign.FillColor = colorBenign
		benign.Offset = -boxWidth/2 - vg.Points(2)

		malignant, err := plotter.NewBoxPlot(boxWidth, 0, plotter.Values(values["M"]))
		if err != nil {
			return err
		}
		malignant.FillColor = colorMalignant
		malignant.Offset = boxWidth/2 + vg.Points(2)

		p := plot.New()
		p.Title.TextStyle.Font.Size = vg.Points(9)
		p.Y.Tick.Label.Font.Size = vg.Points(6)
		p.Add(benign, malignant)
		p.NominalX(abbreviateFeature(analytics.FeatureNames[idx]))
		if k == 0 {
			p.Title.Text = "blue=Benign, orange=Malignant"
		}

		plots[k/cols][k%cols] = p
	}

	img := vgimg.New(16*vg.Inch, 7*vg.Inch)
	dc := draw.New(img)
	tiles := draw.Tiles{
		Rows: rows, Cols: cols,
		PadX: vg.Millimeter * 4, PadY: vg.Millimeter * 4,
		PadTop: vg.Points(4), PadBottom: vg.Points(4),
		PadLeft: vg.Points(4), PadRight: vg.Points(4),
	}

	canvases := plot.Align(plots, tiles, dc)
	for r := range plots {
		for c := range plots[r] {
			if plots[r][c] != nil {
				plots[r][c].Draw(canvases[r][c])
			}
		}
	}

	f, err := os.Create(filepath.Join(outDir, "feature_separation.png"))
	if err != nil {
		return err
	}
	defer f.Close()

	png := vgimg.PngCanvas{Canvas: img}
	_, err = png.WriteTo(f)
	return err
}

type correlationGrid struct {
	matrix [][]float64
}

func (g correlationGrid) Dims() (c, r int)   { return len(g.matrix[0]), len(g.matrix) }
func (g correlationGrid) Z(c, r int) float64 { return g.matrix[r][c] }
func (g correlationGrid) X(c int) float64    { return float64(c) }
func (g correlationGrid) Y(r int) float64    { return float64(r) }
