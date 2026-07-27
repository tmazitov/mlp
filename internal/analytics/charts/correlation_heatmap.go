package charts

import (
	"path/filepath"

	"mlp/internal/analytics"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// CorrelationHeatmap charts the full 30x30 Pearson correlation matrix.
// Clusters of strong correlation (e.g. radius/perimeter/area, or a
// *_mean and its *_worst counterpart) point at redundant inputs worth
// dropping or compressing before training.
//
// Values: ds.CorrelationMatrix() computes the Pearson correlation
// coefficient between every pair of the 30 features (1.0 on the diagonal,
// since a feature is perfectly correlated with itself). Each cell (i, j)
// is colored on a fixed -1..1 blue-white-red scale, so red = strongly
// positively correlated, blue = strongly negatively correlated, white =
// uncorrelated — comparable across the whole image regardless of what the
// actual min/max correlation in this dataset happens to be.
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

// correlationGrid adapts a plain [][]float64 correlation matrix to gonum's
// plotter.GridXYZ interface so it can be rendered as a heatmap.
type correlationGrid struct {
	matrix [][]float64
}

func (g correlationGrid) Dims() (c, r int)   { return len(g.matrix[0]), len(g.matrix) }
func (g correlationGrid) Z(c, r int) float64 { return g.matrix[r][c] }
func (g correlationGrid) X(c int) float64    { return float64(c) }
func (g correlationGrid) Y(r int) float64    { return float64(r) }
