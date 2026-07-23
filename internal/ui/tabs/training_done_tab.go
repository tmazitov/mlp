package tabs

import (
	"fmt"
	"strings"

	"mlp/internal/ui/styles"

	tea "charm.land/bubbletea/v2"
)

// TrainingDoneTabName identifies this tab. Like the process tab, it isn't
// listed in the sidebar — it's reached automatically once training finishes.
const TrainingDoneTabName = "training_done"

const mockWeightsPath = "./checkpoints/model.mlp"

type TrainingDoneTab struct {
	weightsPath string
}

func NewTrainingDoneTab() *TrainingDoneTab {
	return &TrainingDoneTab{
		weightsPath: mockWeightsPath,
	}
}

func (t *TrainingDoneTab) Name() string  { return TrainingDoneTabName }
func (t *TrainingDoneTab) Title() string { return "Done!" }

func (t *TrainingDoneTab) Update(message tea.KeyMsg) tea.Cmd {
	return nil
}

func (t *TrainingDoneTab) View() string {
	var b strings.Builder

	b.WriteString(styles.TabTitleStyle.Render(t.Title()))
	b.WriteString("\n\n")

	b.WriteString("The MLP model was trained successfully!")
	b.WriteString("\n\n")

	link := styles.LinkStyle.Hyperlink("file://" + t.weightsPath).Render(t.weightsPath)
	b.WriteString(styles.DescriptionStyle.Render(fmt.Sprintf("Weights were saved in %s", link)))

	return b.String()
}
