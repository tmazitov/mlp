package tabs

import (
	"fmt"
	"strings"

	"mlp/internal/analytics/charts"
	"mlp/internal/network"
	"mlp/internal/ui/styles"

	"charm.land/bubbles/v2/progress"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

const (
	trainingOutputDir = "training_output"
	modelFilePath     = "model.json"
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
	mascot         Mascot
	animating      bool
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

	// Training is over, so the mascot has nothing left to breathe through —
	// letting it settle also ends the animation tick loop (see Animating).
	if value >= 1 {
		t.animating = false
	}
}

// TickMascot advances the mascot animation by one frame.
func (t *TrainingProcessTab) TickMascot() {
	t.mascot.Tick()
}

// Animating reports whether the mascot should keep moving. The Update loop
// only reschedules MascotTickCmd while this holds, so the animation costs
// nothing once training has finished.
func (t *TrainingProcessTab) Animating() bool {
	return t.animating
}

// StartTraining stores model (created on the training menu tab from the
// submitted form) and runs it in the background. Tab state may only change
// inside Update, so the two goroutines below never touch t directly — they
// report back through program.Send, same as AddLogCmd/UpdateProgressStatusCmd
// do for in-Update callers.
func (t *TrainingProcessTab) StartTraining(msg StartTrainingMsg, program *tea.Program) {
	model, trainSet, valSet := msg.Model, msg.Train, msg.Val
	logs, epochs := msg.Logs, msg.Epochs
	t.model = model
	t.animating = true

	go func() {
		trainLosses := make([]float64, 0, epochs)
		valLosses := make([]float64, 0, epochs)
		trainAcc := make([]float64, 0, epochs)
		valAcc := make([]float64, 0, epochs)

		for stat := range logs {
			trainLosses = append(trainLosses, stat.TrainLoss)
			valLosses = append(valLosses, stat.ValLoss)
			trainAcc = append(trainAcc, stat.TrainAccuracy)
			valAcc = append(valAcc, stat.ValAccuracy)

			program.Send(AddLogMsg{Message: fmt.Sprintf(
				"epoch %d/%d — loss %.4f/%.4f (train/val) — acc %.4f/%.4f (train/val)",
				stat.Epoch+1, epochs, stat.TrainLoss, stat.ValLoss, stat.TrainAccuracy, stat.ValAccuracy,
			)})
			program.Send(UpdateProgressStatusMsg{Value: float64(stat.Epoch+1) / float64(epochs)})
		}

		// logs closes once training finishes (see the goroutine below), so
		// by this point every slice above holds one entry per epoch.
		if lossPath, err := charts.LossCurve(trainLosses, valLosses, trainingOutputDir); err != nil {
			program.Send(AddLogMsg{Message: "loss curve: " + err.Error()})
		} else {
			program.Send(SetLossCurveMsg{Path: lossPath})
		}

		if accPath, err := charts.AccuracyCurve(trainAcc, valAcc, trainingOutputDir); err != nil {
			program.Send(AddLogMsg{Message: "accuracy curve: " + err.Error()})
		} else {
			program.Send(SetAccuracyCurveMsg{Path: accPath})
		}
	}()

	go func() {
		defer close(logs)

		if err := model.Train(trainSet, valSet); err != nil {
			program.Send(AddLogMsg{Message: "training failed: " + err.Error()})
			return
		}

		// The subject asks the training phase to save the model at the end
		// of its run, so prediction has something to load.
		trained := &network.TrainedModel{Network: model, Fields: msg.Fields, Scaler: msg.Scaler}
		if err := trained.Save(modelFilePath); err != nil {
			program.Send(AddLogMsg{Message: "save model: " + err.Error()})
		} else {
			program.Send(AddLogMsg{Message: "model saved to " + modelFilePath})
			program.Send(SetWeightsPathMsg{Path: modelFilePath})
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

	//Logs viewport, with the mascot floating alongside it
	t.logs.SetContentLines(t.logsValues)
	t.logs.GotoBottom()
	b.WriteString(styles.FormLabelStyle.Render("Logs"))
	b.WriteRune('\n')
	b.WriteString(lipgloss.JoinHorizontal(
		lipgloss.Center,
		styles.BoxStyle.Render(t.logs.View()),
		"  ",
		t.mascot.View(),
	))

	return b.String()
}
