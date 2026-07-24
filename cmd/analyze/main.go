// Command analyze generates dataset charts that inform decisions before
// training the MLP (class balance, feature scale, correlated/redundant
// features, and per-feature class separation).
package main

import (
	"flag"
	"fmt"
	"log"

	"mlp/internal/analytics"
	"mlp/internal/analytics/charts"
)

func main() {
	dataPath := flag.String("data", "data.csv", "path to the dataset csv")
	outDir := flag.String("out", "analytics_output", "directory to write charts to")
	flag.Parse()

	if err := run(*dataPath, *outDir); err != nil {
		log.Fatal(err)
	}
}

func run(dataPath, outDir string) error {
	ds, err := analytics.Load(dataPath)
	if err != nil {
		return err
	}

	if err := charts.GenerateAll(ds, outDir); err != nil {
		return err
	}

	fmt.Printf("charts saved to %s\n", outDir)
	return nil
}
