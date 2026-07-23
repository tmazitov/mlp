package tabs

import (
	"mlp/internal/ui/styles"
	"strings"

	tea "charm.land/bubbletea/v2"
)

type TrainingMenuTab struct {
	form *TrainingForm
}

func NewTrainingMenuTab() *TrainingMenuTab {
	return &TrainingMenuTab{
		form: NewTrainingForm(),
	}
}

func (t TrainingMenuTab) Name() string  { return "training_menu" }
func (t TrainingMenuTab) Title() string { return "Training" }

func (t TrainingMenuTab) Update(message tea.KeyMsg) tea.Cmd {
	return t.form.Update(message)
}

func (t TrainingMenuTab) View() string {
	var contentBuilder strings.Builder

	contentBuilder.WriteString(styles.TabTitleStyle.Render(t.Title()))
	contentBuilder.WriteRune('\n')
	contentBuilder.WriteString(t.form.View())

	return contentBuilder.String()
}
