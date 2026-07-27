// Package charts renders PNG charts from an analytics.Dataset that are
// meant to inform decisions before training the MLP: whether the classes
// are balanced, whether inputs need standardizing, which features are
// redundant, and which features actually separate the two classes.
package charts

import (
	"fmt"
	"image/color"
	"os"

	"mlp/internal/analytics"
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
