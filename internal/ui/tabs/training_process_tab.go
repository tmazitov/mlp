package tabs

import (
	"fmt"
	"strings"

	"mlp/internal/analytics"
	"mlp/internal/analytics/log"
	"mlp/internal/network"
	"mlp/internal/ui/styles"

	"charm.land/bubbles/v2/progress"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
)

// AddLogMsg appends a line to the training process log.
type AddLogMsg struct {
	Message string
}

// AddLogCmd wraps a log line as a tea.Cmd. Send the resulting command from
// Update/Init (or the message itself via *tea.Program.Send from another
// goroutine) to append to the log without touching tab state directly —
// state only ever changes inside the Update loop, so it stays race-free.
func AddLogCmd(message string) tea.Cmd {
	return func() tea.Msg {
		return AddLogMsg{Message: message}
	}
}

// UpdateProgressStatusMsg sets the training process progress bar (0-1).
type UpdateProgressStatusMsg struct {
	Value float64
}

// UpdateProgressStatusCmd wraps a progress value as a tea.Cmd, mirroring
// AddLogCmd.
func UpdateProgressStatusCmd(value float64) tea.Cmd {
	return func() tea.Msg {
		return UpdateProgressStatusMsg{Value: value}
	}
}

type TrainingProcessTab struct {
	progress       progress.Model
	logs           viewport.Model
	progressStatus float64
	logsValues     []string
	model          *network.MLP
}

func NewTrainingProcessTab() *TrainingProcessTab {
	prog := progress.New(progress.WithColors(styles.PrimaryColor[700], styles.PrimaryColor[400]))
	prog.SetWidth(40)

	logs := viewport.New(viewport.WithWidth(50), viewport.WithHeight(10))

	return &TrainingProcessTab{
		progress: prog,
		logs:     logs,
	}
}

func (t *TrainingProcessTab) Name() string  { return "training_process" }
func (t *TrainingProcessTab) Title() string { return "Process" }

func (t *TrainingProcessTab) AddLog(logMessage string) {
	t.logsValues = append(t.logsValues, logMessage)
}
func (t *TrainingProcessTab) UpdateProgressStatus(value float64) {
	t.progressStatus = value
}

// StartTraining stores model (created on the training menu tab from the
// submitted form) and runs it in the background. Tab state may only change
// inside Update, so the two goroutines below never touch t directly — they
// report back through program.Send, same as AddLogCmd/UpdateProgressStatusCmd
// do for in-Update callers.
func (t *TrainingProcessTab) StartTraining(model *network.MLP, dataset *analytics.Dataset, logs chan log.TrainingStat, epochs int, program *tea.Program) {
	t.model = model

	go func() {
		for stat := range logs {
			program.Send(AddLogMsg{Message: fmt.Sprintf("epoch %d/%d — avg loss %.4f", stat.Epoch+1, epochs, stat.AverageLoss)})
			program.Send(UpdateProgressStatusMsg{Value: float64(stat.Epoch+1) / float64(epochs)})
		}
	}()

	go func() {
		defer close(logs)

		if err := model.Train(dataset); err != nil {
			program.Send(AddLogMsg{Message: "training failed: " + err.Error()})
			return
		}

		program.Send(UpdateProgressStatusMsg{Value: 1})
		program.Send(SwitchTabMsg{TabName: TrainingDoneTabName})
	}()
}

func (t *TrainingProcessTab) Update(message tea.KeyMsg) tea.Cmd {
	var cmd tea.Cmd
	t.logs, cmd = t.logs.Update(message)
	return cmd
}

func (t *TrainingProcessTab) View() string {
	var b strings.Builder

	//Title
	b.WriteString(styles.TabTitleStyle.Render(t.Title()))
	b.WriteRune('\n')

	//Progress bar
	status := "waiting to start"
	if len(t.logsValues) > 0 {
		status = t.logsValues[len(t.logsValues)-1]
	}
	subtitle := fmt.Sprintf("%s — %.0f%%", status, t.progressStatus*100)
	b.WriteString(styles.FormLabelStyle.Render(subtitle))
	b.WriteRune('\n')
	b.WriteString(t.progress.ViewAs(t.progressStatus))
	b.WriteString("\n\n")

	//Logs viewport
	t.logs.SetContentLines(t.logsValues)
	t.logs.GotoBottom()
	b.WriteString(styles.FormLabelStyle.Render("Logs"))
	b.WriteRune('\n')
	b.WriteString(styles.BoxStyle.Render(t.logs.View()))

	return b.String()
}
