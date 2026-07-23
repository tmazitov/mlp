package tabs

import (
	"fmt"
	"strings"

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
