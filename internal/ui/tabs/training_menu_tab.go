package tabs

import (
	"mlp/internal/ui/styles"
	"strings"

	tea "charm.land/bubbletea/v2"
)

type TrainingMenuTab struct{}

func NewTrainingMenuTab() *TrainingMenuTab {

	return &TrainingMenuTab{}
}

func (t TrainingMenuTab) Name() string  { return "training_menu" }
func (t TrainingMenuTab) Title() string { return "Training" }

func (t TrainingMenuTab) Update(message tea.KeyMsg) tea.Cmd {
	return nil
}

func (t TrainingMenuTab) View() string {
	var contentBuilder strings.Builder

	contentBuilder.WriteString(styles.TabTitleStyle.Render(t.Title()))

	return contentBuilder.String()
}
