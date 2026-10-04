package tabs

import (
	"fmt"
	"strings"
	"sync"

	"mlp/internal/analytics/charts"
	"mlp/internal/analytics/log"
	"mlp/internal/ui/styles"

	tea "charm.land/bubbletea/v2"
)

const comparisonOutputDir = "training_output"

// ModelProgressMsg reports one epoch of one model in a comparison run.
type ModelProgressMsg struct {
	Index int
	Stat  log.TrainingStat
}

// ModelFinishedMsg reports that one model's run ended, with every epoch it
// recorded so the comparison charts can be drawn once all of them are in.
type ModelFinishedMsg struct {
	Index   int
	History []log.TrainingStat
	Err     error
}

// RunModelsMsg asks the UI to start the selected runs. The tab cannot do
// it itself because the goroutines report back through *tea.Program, which
// only the UI holds — and the UI must not decide when enter means "run",
// since only the tab knows whether the new-configuration form is open.
type RunModelsMsg struct{}

func RunModelsCmd() tea.Cmd {
	return func() tea.Msg { return RunModelsMsg{} }
}

// ComparisonReadyMsg carries the charts drawn once every selected model
// has finished.
type ComparisonReadyMsg struct {
	LossPath     string
	AccuracyPath string
	Err          error
}

type runState int

const (
	runIdle runState = iota
	runRunning
	runDone
	runFailed
)

// modelEntry is one configuration in the list, plus whatever the last run
// of it produced.
type modelEntry struct {
	name     string
	config   TrainingConfig
	selected bool

	state        runState
	earlyStopped bool
	bestEpoch    int
	epoch        int
	stat         log.TrainingStat
	history      []log.TrainingStat
	err          error

	historyPath string
}

type ModelsTab struct {
	entries []modelEntry
	hovered int

	// form is non-nil while a new configuration is being entered, which is
	// also what tells Update to send keys there instead of the list.
	form *TrainingForm

	errorMsg     string
	running      bool
	lossChart    string
	accuracyChrt string
}

func NewModelsTab() *ModelsTab {
	// Seeded with configurations worth putting side by side rather than an
	// empty list: the three optimizers at the learning rate each wants.
	base := func(name string, layers []uint, lr float64, optimizer string) modelEntry {
		return modelEntry{
			name: name,
			config: TrainingConfig{
				layers:       layers,
				epochs:       500,
				lossFunc:     "cross-entropy",
				batchSize:    "16",
				learningRate: lr,
				optimizer:    optimizer,
			},
			selected: true,
		}
	}

	return &ModelsTab{
		entries: []modelEntry{
			base("sgd 8x8", []uint{8, 8}, 0.05, "sgd"),
			base("nesterov 8x8", []uint{8, 8}, 0.05, "nesterov"),
			base("rmsprop 8x8", []uint{8, 8}, 0.001, "rmsprop"),
		},
	}
}

func (t *ModelsTab) Name() string  { return "models" }
func (t *ModelsTab) Title() string { return "Models" }

func (t *ModelsTab) Update(message tea.KeyMsg) tea.Cmd {
	// While the form is open it owns every key, so typing "a" or a space
	// into a field cannot also add or toggle a list entry.
	if t.form != nil {
		cmd := t.form.Update(message)

		if message.String() == "esc" {
			t.form = nil
			return cmd
		}

		if cfg, ok := t.form.TakeSubmission(); ok {
			t.entries = append(t.entries, modelEntry{
				name:     describeConfig(cfg),
				config:   cfg,
				selected: true,
			})
			t.form = nil
		}
		return cmd
	}

	switch message.String() {
	case "up", "k":
		if len(t.entries) > 0 {
			t.hovered = (t.hovered - 1 + len(t.entries)) % len(t.entries)
		}
	case "down", "j":
		if len(t.entries) > 0 {
			t.hovered = (t.hovered + 1) % len(t.entries)
		}
	case "space":
		if len(t.entries) > 0 {
			t.entries[t.hovered].selected = !t.entries[t.hovered].selected
		}
	case "a":
		t.form = NewTrainingForm()
	case "d":
		if len(t.entries) > 0 && !t.running {
			t.entries = append(t.entries[:t.hovered], t.entries[t.hovered+1:]...)
			if t.hovered >= len(t.entries) && t.hovered > 0 {
				t.hovered--
			}
		}
	case "enter":
		return RunModelsCmd()
	}

	return nil
}

// Run trains every selected configuration at once, each in its own
// goroutine with its own stats channel, reporting back through
// program.Send. Nothing here touches tab state directly: that only changes
// inside Update, which is what keeps the runs from racing the renderer.
func (t *ModelsTab) Run(program *tea.Program) {
	if t.running {
		return
	}

	selected := make([]int, 0, len(t.entries))
	for i, entry := range t.entries {
		if entry.selected {
			selected = append(selected, i)
		}
	}
	if len(selected) == 0 {
		t.errorMsg = "select at least one model with space"
		return
	}

	data, err := loadTrainingData()
	if err != nil {
		t.errorMsg = err.Error()
		return
	}

	t.errorMsg = ""
	t.running = true
	t.lossChart, t.accuracyChrt = "", ""
	for _, i := range selected {
		t.entries[i].state = runRunning
		t.entries[i].epoch = 0
		t.entries[i].history = nil
		t.entries[i].err = nil
	}

	var wg sync.WaitGroup
	for _, index := range selected {
		wg.Add(1)

		go func(index int, cfg TrainingConfig) {
			defer wg.Done()

			model, logs, err := buildModel(cfg)
			if err != nil {
				program.Send(ModelFinishedMsg{Index: index, Err: err})
				return
			}

			// One collector per model: the stats channel is per-run, and
			// its index is what lets the UI tell the rows apart.
			history := make([]log.TrainingStat, 0, cfg.epochs)
			collected := make(chan []log.TrainingStat, 1)
			go func() {
				for stat := range logs {
					// The early-stop marker is a verdict, not an epoch's
					// worth of data, so it stays out of the curve.
					if stat.EarlyStopped {
						program.Send(ModelProgressMsg{Index: index, Stat: stat})
						continue
					}
					history = append(history, stat)
					program.Send(ModelProgressMsg{Index: index, Stat: stat})
				}
				collected <- history
			}()

			trainErr := model.Train(data.train, data.val)
			close(logs)

			program.Send(ModelFinishedMsg{Index: index, History: <-collected, Err: trainErr})
		}(index, t.entries[index].config)
	}

	// Drawing the comparison needs every run's history, so it waits for
	// all of them rather than redrawing after each.
	go func() {
		wg.Wait()
		program.Send(ComparisonReadyMsg{})
	}()
}

// ApplyProgress and the two below are the Update-side half of Run: they are
// the only things that write to the entries.
func (t *ModelsTab) ApplyProgress(msg ModelProgressMsg) {
	if msg.Index < 0 || msg.Index >= len(t.entries) {
		return
	}
	if msg.Stat.EarlyStopped {
		t.entries[msg.Index].earlyStopped = true
		t.entries[msg.Index].bestEpoch = msg.Stat.BestEpoch + 1
		return
	}
	t.entries[msg.Index].epoch = msg.Stat.Epoch + 1
	t.entries[msg.Index].stat = msg.Stat
}

func (t *ModelsTab) ApplyFinished(msg ModelFinishedMsg) {
	if msg.Index < 0 || msg.Index >= len(t.entries) {
		return
	}

	entry := &t.entries[msg.Index]
	entry.history = msg.History
	entry.err = msg.Err
	if msg.Err != nil {
		entry.state = runFailed
		return
	}
	entry.state = runDone
}

// BuildComparison draws the charts once every run has reported in. It is
// called from Update, so it reads the histories without racing the
// goroutines that filled them.
func (t *ModelsTab) BuildComparison() ComparisonReadyMsg {
	t.running = false

	var losses, accuracies []charts.LabeledSeries
	for _, entry := range t.entries {
		if entry.state != runDone || len(entry.history) == 0 {
			continue
		}

		valLoss := make([]float64, len(entry.history))
		valAcc := make([]float64, len(entry.history))
		for i, stat := range entry.history {
			valLoss[i] = stat.Val.Loss
			valAcc[i] = stat.Val.Accuracy()
		}

		losses = append(losses, charts.LabeledSeries{Name: entry.name, Values: valLoss})
		accuracies = append(accuracies, charts.LabeledSeries{Name: entry.name, Values: valAcc})
	}

	if len(losses) == 0 {
		return ComparisonReadyMsg{Err: fmt.Errorf("no model finished successfully")}
	}

	// One history per model, named after it: the comparison charts show
	// the shape, these keep the numbers behind each line.
	for i := range t.entries {
		if t.entries[i].state != runDone || len(t.entries[i].history) == 0 {
			continue
		}
		path, err := log.WriteHistory(t.entries[i].history, t.entries[i].name, comparisonOutputDir)
		if err != nil {
			return ComparisonReadyMsg{Err: err}
		}
		t.entries[i].historyPath = path
	}

	lossPath, err := charts.ComparisonCurve(losses,
		"Validation loss by epoch", "validation loss", "comparison_loss.png", comparisonOutputDir)
	if err != nil {
		return ComparisonReadyMsg{Err: err}
	}

	accuracyPath, err := charts.ComparisonCurve(accuracies,
		"Validation accuracy by epoch", "validation accuracy", "comparison_accuracy.png", comparisonOutputDir)
	if err != nil {
		return ComparisonReadyMsg{Err: err}
	}

	return ComparisonReadyMsg{LossPath: lossPath, AccuracyPath: accuracyPath}
}

func (t *ModelsTab) ApplyComparison(msg ComparisonReadyMsg) {
	t.running = false
	if msg.Err != nil {
		t.errorMsg = msg.Err.Error()
		return
	}
	t.lossChart, t.accuracyChrt = msg.LossPath, msg.AccuracyPath
}

// Running reports whether any run is still going, so the Update loop knows
// whether a comparison is pending.
func (t *ModelsTab) Running() bool { return t.running }

// describeConfig names an entry after what makes it different from the
// others, which is what you are reading the list for.
func describeConfig(cfg TrainingConfig) string {
	sizes := make([]string, len(cfg.layers))
	for i, size := range cfg.layers {
		sizes[i] = fmt.Sprint(size)
	}
	return fmt.Sprintf("%s %s lr=%g", cfg.optimizer, strings.Join(sizes, "x"), cfg.learningRate)
}

func (t *ModelsTab) View() string {
	var b strings.Builder

	b.WriteString(styles.TabTitleStyle.Render(t.Title()))
	b.WriteRune('\n')
	b.WriteString(styles.DescriptionStyle.Render(
		"Trains the selected configurations at once and charts them together."))
	b.WriteString("\n\n")

	if t.form != nil {
		b.WriteString(styles.FormLabelFocusedStyle.Render("New configuration"))
		b.WriteRune('\n')
		b.WriteString(t.form.View())
		b.WriteRune('\n')
		b.WriteString(styles.FormHintStyle.Render("esc: cancel"))
		return b.String()
	}

	for i, entry := range t.entries {
		cursor, checkbox := "  ", "[ ]"
		if i == t.hovered {
			cursor = "> "
		}
		if entry.selected {
			checkbox = "[x]"
		}

		line := fmt.Sprintf("%s%s %-22s %s", cursor, checkbox, entry.name, entry.statusText())

		style := styles.FormLabelStyle
		if i == t.hovered {
			style = styles.FormLabelFocusedStyle
		}
		b.WriteString(style.Render(line))
		b.WriteRune('\n')
	}

	if t.errorMsg != "" {
		b.WriteRune('\n')
		b.WriteString(styles.FormErrorStyle.Render("Error: " + t.errorMsg))
		b.WriteRune('\n')
	}

	if t.lossChart != "" {
		b.WriteRune('\n')
		b.WriteString(styles.FormLabelFocusedStyle.Render("Comparison"))
		b.WriteRune('\n')
		b.WriteString(styles.DescriptionStyle.Render(fileLink(t.lossChart)))
		b.WriteRune('\n')
		b.WriteString(styles.DescriptionStyle.Render(fileLink(t.accuracyChrt)))
		b.WriteRune('\n')
	}

	b.WriteRune('\n')
	b.WriteString(styles.FormHintStyle.Render(
		"↑/↓: move • space: select • a: add • d: delete • enter: run selected"))

	return b.String()
}

// statusText is the right-hand column of a row: what this configuration is
// doing, or what its last run came to.
func (e modelEntry) statusText() string {
	switch e.state {
	case runRunning:
		return fmt.Sprintf("epoch %d — val loss %.4f acc %.4f",
			e.epoch, e.stat.Val.Loss, e.stat.Val.Accuracy())
	case runDone:
		best := e.stat.Val.Loss
		for _, stat := range e.history {
			if stat.Val.Loss < best {
				best = stat.Val.Loss
			}
		}
		if e.earlyStopped {
			return fmt.Sprintf("stopped at epoch %d, best %d — val loss %.4f acc %.4f",
				e.epoch, e.bestEpoch, best, e.stat.Val.Accuracy())
		}
		return fmt.Sprintf("done — val loss %.4f (best %.4f) acc %.4f",
			e.stat.Val.Loss, best, e.stat.Val.Accuracy())
	case runFailed:
		return "failed: " + e.err.Error()
	default:
		return ""
	}
}
