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

// SetLossCurveMsg sets the path to the just-generated training-loss chart
// on the Done tab, once StartTraining has finished writing it.
type SetLossCurveMsg struct {
	Path string
}

// SetWeightsPathMsg records where the training phase saved the model.
type SetWeightsPathMsg struct {
	Path string
}

// SetAccuracyCurveMsg sets the path to the just-generated accuracy chart
// on the Done tab, once StartTraining has finished writing it.
type SetAccuracyCurveMsg struct {
	Path string
}

// SetHistoryPathMsg records where this run's per-epoch metrics were
// written.
type SetHistoryPathMsg struct {
	Path string
}

type TrainingDoneTab struct {
	weightsPath       string
	lossCurvePath     string
	accuracyCurvePath string
	historyPath       string
}

// SetHistoryPath records where this run's metric history was saved.
func (t *TrainingDoneTab) SetHistoryPath(path string) {
	t.historyPath = path
}

func NewTrainingDoneTab() *TrainingDoneTab {
	return &TrainingDoneTab{}
}

// SetWeightsPath records where this run's model was written.
func (t *TrainingDoneTab) SetWeightsPath(path string) {
	t.weightsPath = path
}

func (t *TrainingDoneTab) Name() string  { return TrainingDoneTabName }
func (t *TrainingDoneTab) Title() string { return "Done!" }

// SetLossCurvePath records where the loss chart for this run was saved, so
// View can link to it.
func (t *TrainingDoneTab) SetLossCurvePath(path string) {
	t.lossCurvePath = path
}

// SetAccuracyCurvePath records where the accuracy chart for this run was
// saved, so View can link to it.
func (t *TrainingDoneTab) SetAccuracyCurvePath(path string) {
	t.accuracyCurvePath = path
}

func (t *TrainingDoneTab) Update(message tea.KeyMsg) tea.Cmd {
	return nil
}

func (t *TrainingDoneTab) View() string {
	var b strings.Builder

	b.WriteString(styles.TabTitleStyle.Render(t.Title()))
	b.WriteString("\n\n")

	b.WriteString("The MLP model was trained successfully!")
	b.WriteString("\n\n")

	if t.lossCurvePath != "" {
		lossLink := fileLink(t.lossCurvePath)
		b.WriteString(styles.DescriptionStyle.Render(fmt.Sprintf("Loss curve saved in %s", lossLink)))
		b.WriteString("\n\n")
	}

	if t.accuracyCurvePath != "" {
		accLink := fileLink(t.accuracyCurvePath)
		b.WriteString(styles.DescriptionStyle.Render(fmt.Sprintf("Accuracy curve saved in %s", accLink)))
		b.WriteString("\n\n")
	}

	if t.historyPath != "" {
		historyLink := fileLink(t.historyPath)
		b.WriteString(styles.DescriptionStyle.Render(fmt.Sprintf("Metric history saved in %s", historyLink)))
		b.WriteString("\n\n")
	}

	if t.weightsPath != "" {
		weightsLink := fileLink(t.weightsPath)
		b.WriteString(styles.DescriptionStyle.Render(fmt.Sprintf("Model saved in %s", weightsLink)))
	}

	return b.String()
}
