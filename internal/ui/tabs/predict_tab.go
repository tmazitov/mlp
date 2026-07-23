package tabs

import (
	"mlp/internal/ui/styles"
	"strings"

	tea "charm.land/bubbletea/v2"
)

type PredictTab struct{}

func NewPredictTab() *PredictTab {

	return &PredictTab{}
}

func (t PredictTab) Name() string  { return "predict_menu" }
func (t PredictTab) Title() string { return "Predict" }

func (t PredictTab) Update(message tea.KeyMsg) tea.Cmd {
	return nil
}

func (t PredictTab) View() string {
	var contentBuilder strings.Builder

	contentBuilder.WriteString(styles.TabTitleStyle.Render(t.Title()))

	return contentBuilder.String()
}
