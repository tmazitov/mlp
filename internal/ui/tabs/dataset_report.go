package tabs

import (
	"fmt"
	"sort"
	"strings"

	"mlp/internal/analytics"
	"mlp/internal/ui/styles"
)

const classBarWidth = 40

func renderDatasetReport(ds *analytics.Dataset) string {
	var b strings.Builder

	b.WriteString(styles.FormLabelFocusedStyle.Render(fmt.Sprintf("%d rows", len(ds.Rows))))
	b.WriteString("\n\n")

	renderClassBalance(&b, ds)
	renderFeatureSeparation(&b, ds)
	renderTopCorrelations(&b, ds)

	return b.String()
}

func renderClassBalance(b *strings.Builder, ds *analytics.Dataset) {
	counts := ds.ClassCounts()
	total := len(ds.Rows)

	b.WriteString(styles.FormLabelFocusedStyle.Render("Class balance"))
	b.WriteRune('\n')

	diagnoses := make([]string, 0, len(counts))
	for diagnosis := range counts {
		diagnoses = append(diagnoses, diagnosis)
	}
	sort.Strings(diagnoses)

	for _, diagnosis := range diagnoses {
		count := counts[diagnosis]
		pct := float64(count) / float64(total) * 100
		bar := strings.Repeat("█", count*classBarWidth/total)
		fmt.Fprintf(b, "%-2s %4d (%5.1f%%) %s\n", diagnosis, count, pct, bar)
	}
	b.WriteString("\n")
}

func renderFeatureSeparation(b *strings.Builder, ds *analytics.Dataset) {
	statsByClass := ds.FeatureStatsByDiagnosis()

	b.WriteString(styles.FormLabelFocusedStyle.Render("Feature separation (mean group), by diagnosis"))
	b.WriteRune('\n')
	fmt.Fprintf(b, "%-24s %10s %10s %10s %10s\n", "feature", "mean(B)", "mean(M)", "std(B)", "std(M)")

	for i, name := range analytics.FeatureNames {
		if !strings.HasSuffix(name, "_mean") {
			continue
		}
		bStats := statsByClass["B"][i]
		mStats := statsByClass["M"][i]
		fmt.Fprintf(b, "%-24s %10.3f %10.3f %10.3f %10.3f\n", name, bStats.Mean, mStats.Mean, bStats.StdDev, mStats.StdDev)
	}
	b.WriteString("\n")
}

func renderTopCorrelations(b *strings.Builder, ds *analytics.Dataset) {
	b.WriteString(styles.FormLabelFocusedStyle.Render("Highly correlated feature pairs (|r| >= 0.9)"))
	b.WriteRune('\n')

	for _, pair := range ds.TopCorrelations(0.9) {
		fmt.Fprintf(b, "%-24s <-> %-24s  r=%.3f\n", pair.FeatureA, pair.FeatureB, pair.Correlation)
	}
}
