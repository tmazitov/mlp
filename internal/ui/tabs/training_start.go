package tabs

import (
	"mlp/internal/analytics"
	"mlp/internal/analytics/log"
	"mlp/internal/network"

	tea "charm.land/bubbletea/v2"
)

// StartTrainingMsg carries a model built from the submitted training form
// (see training_menu_tab.go) over to the process tab, which owns the
// goroutines that run it and relay its progress back into the TUI.
type StartTrainingMsg struct {
	Model  *network.MLP
	Train  *analytics.Dataset
	Val    *analytics.Dataset
	Logs   chan log.TrainingStat
	Epochs int
}

func StartTrainingCmd(model *network.MLP, trainSet, valSet *analytics.Dataset, logs chan log.TrainingStat, epochs int) tea.Cmd {
	return func() tea.Msg {
		return StartTrainingMsg{Model: model, Train: trainSet, Val: valSet, Logs: logs, Epochs: epochs}
	}
}
