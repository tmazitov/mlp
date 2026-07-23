package app

import (
	"fmt"
	"time"

	"mlp/internal/ui/tabs"

	tea "charm.land/bubbletea/v2"
)

// simulateTraining stands in for a real trainer: it feeds mock log lines and
// progress updates to the running program over time. It runs in its own
// goroutine, so updates are delivered via program.Send rather than by
// calling tab methods directly — tab state only ever changes inside the
// Update loop, which keeps it race-free with rendering.
func simulateTraining(program *tea.Program) {
	const totalSteps = 100

	for i := range totalSteps {
		time.Sleep(200 * time.Millisecond)
		program.Send(tabs.AddLogMsg{Message: fmt.Sprintf("Log %d: something big happen..", i)})
		program.Send(tabs.UpdateProgressStatusMsg{Value: float64(i) / totalSteps})
	}

	program.Send(tabs.UpdateProgressStatusMsg{Value: 1})
	program.Send(tabs.SwitchTabMsg{TabName: tabs.TrainingDoneTabName})
}
