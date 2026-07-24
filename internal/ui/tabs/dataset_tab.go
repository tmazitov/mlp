package tabs

import (
	"fmt"
	"path/filepath"
	"strings"

	"mlp/internal/analytics"
	"mlp/internal/analytics/charts"
	"mlp/internal/ui/styles"

	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

const (
	datasetCSVPath  = "data.csv"
	chartsOutputDir = "analytics_output"
)

type chartEntry struct {
	key   string
	label string
	path  string
}

// styles.BoxStyle: RoundedBorder (1 char each side) + Padding(1, 2, 0, 2).
const (
	boxOverheadHorizontal = 6 // border l/r(2) + padding l/r(4)
	boxOverheadVertical   = 3 // border t/b(2) + padding t/b(1)
)

// menuColumnWidth is the chart-menu column's fixed OUTER width (i.e. the
// value passed to Style.Width, border+padding included) — its content is
// just a handful of short labels, so it doesn't need to grow with the
// terminal. It's wide enough for "[3] Correlation heatmap" (24 chars) plus
// the box's own horizontal overhead.
const menuColumnWidth = 24 + boxOverheadHorizontal

type DatasetTab struct {
	viewport  viewport.Model
	charts    []chartEntry
	statusMsg string
}

func NewDatasetTab() *DatasetTab {
	vp := viewport.New(viewport.WithWidth(72), viewport.WithHeight(25))

	tab := &DatasetTab{viewport: vp}

	ds, err := analytics.Load(datasetCSVPath)
	if err != nil {
		vp.SetContent(styles.FormErrorStyle.Render(fmt.Sprintf("failed to load %s: %v", datasetCSVPath, err)))
		tab.viewport = vp
		return tab
	}
	vp.SetContent(renderDatasetReport(ds))
	tab.viewport = vp

	if err := charts.GenerateAll(ds, chartsOutputDir); err != nil {
		tab.statusMsg = fmt.Sprintf("failed to generate charts: %v", err)
		return tab
	}
	tab.charts = []chartEntry{
		{key: "1", label: "Class balance", path: filepath.Join(chartsOutputDir, "class_balance.png")},
		{key: "2", label: "Feature scales", path: filepath.Join(chartsOutputDir, "feature_scales.png")},
		{key: "3", label: "Correlation heatmap", path: filepath.Join(chartsOutputDir, "correlation_heatmap.png")},
		{key: "4", label: "Feature separation", path: filepath.Join(chartsOutputDir, "feature_separation.png")},
	}

	return tab
}

func (t *DatasetTab) Name() string  { return "dataset" }
func (t *DatasetTab) Title() string { return "Dataset" }

// SetSize adapts the report viewport (the wider of the two side-by-side
// boxes) to the actual available space instead of a hardcoded size, so the
// tab no longer overflows (or leaves dead space in) the outer column box.
// width/height is the content area MainWindow was given; boxOverhead*
// accounts for that column's own border+padding and, separately, each of
// the two inner boxes' border+padding — both use styles.BoxStyle — and
// titleLines/statusLines account for the tab's other chrome.
func (t *DatasetTab) SetSize(width, height int) {
	const (
		titleLines  = 1
		statusLines = 1 // reserved even when currently empty, since Update can set it later
	)

	// width/height come in as the outer column's OWN content budget (i.e.
	// already inside that column's border+padding), so only one
	// boxOverheadHorizontal/Vertical applies here, then a second one for
	// the report box nested inside it.
	contentWidth := width - boxOverheadHorizontal
	contentHeight := height - boxOverheadVertical - titleLines - statusLines

	viewportWidth := contentWidth - menuColumnWidth - boxOverheadHorizontal
	viewportHeight := contentHeight - boxOverheadVertical

	if viewportWidth < 20 {
		viewportWidth = 20
	}
	if viewportHeight < 3 {
		viewportHeight = 3
	}

	t.viewport.SetWidth(viewportWidth)
	t.viewport.SetHeight(viewportHeight)
}

func (t *DatasetTab) Update(message tea.KeyMsg) tea.Cmd {
	for _, c := range t.charts {
		if message.String() != c.key {
			continue
		}
		if err := openInSystemViewer(c.path); err != nil {
			t.statusMsg = fmt.Sprintf("failed to open %s: %v", c.label, err)
		} else {
			t.statusMsg = fmt.Sprintf("opened %s (%s)", c.label, c.path)
		}
		return nil
	}

	var cmd tea.Cmd
	t.viewport, cmd = t.viewport.Update(message)
	return cmd
}

func (t *DatasetTab) View() string {
	var b strings.Builder

	b.WriteString(styles.TabTitleStyle.Render(t.Title()))
	b.WriteRune('\n')

	b.WriteString(lipgloss.JoinHorizontal(
		lipgloss.Top,
		styles.BoxStyle.Width(menuColumnWidth).Render(t.paddedChartMenu()),
		styles.BoxStyle.Width(t.viewport.Width()+boxOverheadHorizontal).Render(t.viewport.View()),
	))

	return b.String()
}

// paddedChartMenu pads the (short) chart menu with blank lines to match the
// report viewport's line count, so both boxes in the row auto-size to the
// same total height instead of the menu box ending early.
func (t *DatasetTab) paddedChartMenu() string {
	menu := renderChartMenu(t.charts)
	for lines := strings.Count(menu, "\n") + 1; lines < t.viewport.Height(); lines++ {
		menu += "\n"
	}
	return menu
}

func renderChartMenu(entries []chartEntry) string {
	var b strings.Builder
	b.WriteString("Open chart:")
	for _, e := range entries {
		b.WriteRune('\n')
		fmt.Fprintf(&b, "  [%s] %s", e.key, e.label)
	}
	return b.String()
}
