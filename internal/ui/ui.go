package ui

import (
	"mlp/internal/ui/styles"
	"mlp/internal/ui/tabs"
	"mlp/internal/ui/views"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

type UI struct {
	menuSideBar        *views.MenuSideBar
	mainWindow         *views.MainWindow
	trainingProcessTab *tabs.TrainingProcessTab
	trainingDoneTab    *tabs.TrainingDoneTab
	selectedColumn     int
	width              int
	height             int
	program            *tea.Program
}

// SetProgram gives the UI a handle to the running program, so it can hand it
// to background goroutines (started e.g. once training begins) that need to
// send messages back in from outside the Update loop. Call it once, right
// after tea.NewProgram, before Run.
func (u *UI) SetProgram(program *tea.Program) {
	u.program = program
}

func NewUI() *UI {

	trainingMenuTab := tabs.NewTrainingMenuTab()
	trainingProcessTab := tabs.NewTrainingProcessTab()
	trainingDoneTab := tabs.NewTrainingDoneTab()
	predictTab := tabs.NewPredictTab()
	datasetTab := tabs.NewDatasetTab()
	splitTab := tabs.NewSplitTab()

	// allTabs is every tab MainWindow can display, including ones hidden
	// from the sidebar. menuTabs is only the subset shown in the sidebar —
	// trainingProcessTab and trainingDoneTab are reachable only by
	// submitting the training form and letting training run to completion.
	allTabs := []tabs.Tab{datasetTab, splitTab, trainingMenuTab, trainingProcessTab, trainingDoneTab, predictTab}
	menuTabs := []tabs.Tab{datasetTab, splitTab, trainingMenuTab, predictTab}

	var ui *UI = &UI{
		mainWindow:         views.NewMainWindow(allTabs),
		trainingProcessTab: trainingProcessTab,
		trainingDoneTab:    trainingDoneTab,
		selectedColumn:     0,
	}

	ui.menuSideBar = views.NewMenuSideBar(menuTabs)

	return ui
}

// Init returns an initial commands for the app
func (u UI) Init() tea.Cmd {
	return nil // no init command
}

func (u UI) updateComponent(message tea.KeyMsg) tea.Cmd {

	var cmd tea.Cmd

	switch u.selectedColumn {
	case 0:
		cmd = u.menuSideBar.Update(message)
	case 1:
		cmd = u.mainWindow.Update(message)
	}
	return cmd
}

// Update handlers incoming messages and updates the model
func (u UI) Update(message tea.Msg) (tea.Model, tea.Cmd) {

	var cmd tea.Cmd

	switch message := message.(type) {
	case tea.WindowSizeMsg:
		u.width = message.Width
		u.height = message.Height
		u.mainWindow.SetSize(int(float32(u.width)*0.75), u.height)
	case tabs.SwitchTabMsg:
		u.mainWindow.SetCurrentTab(message.TabName)
		u.selectedColumn = 1
	case tabs.AddLogMsg:
		u.trainingProcessTab.AddLog(message.Message)
	case tabs.UpdateProgressStatusMsg:
		u.trainingProcessTab.UpdateProgressStatus(message.Value)
	case tabs.StartTrainingMsg:
		u.trainingProcessTab.StartTraining(message, u.program)
		// Kick off the mascot animation loop; it keeps itself alive below
		// for as long as training runs.
		cmd = tabs.MascotTickCmd()
	case tabs.MascotTickMsg:
		if u.trainingProcessTab.Animating() {
			u.trainingProcessTab.TickMascot()
			cmd = tabs.MascotTickCmd()
		}
	case tabs.SetLossCurveMsg:
		u.trainingDoneTab.SetLossCurvePath(message.Path)
	case tabs.SetAccuracyCurveMsg:
		u.trainingDoneTab.SetAccuracyCurvePath(message.Path)
	case tabs.SetWeightsPathMsg:
		u.trainingDoneTab.SetWeightsPath(message.Path)
	case tea.KeyMsg:
		switch message.String() {

		case "q", "ctrl+c":
			return u, tea.Quit

		case "right":
			u.selectedColumn = min(1, u.selectedColumn+1)

		case "left":
			u.selectedColumn = max(0, u.selectedColumn-1)

		default:
			cmd = u.updateComponent(message)
		}
	}

	return u, cmd
}

// View renders the current state
func (u UI) View() tea.View {

	columnStyle := styles.BoxStyle.Height(u.height)
	selectedColumnStyle := styles.SelectedBoxStyle.Height(u.height)

	views := []string{
		u.menuSideBar.View(),
		u.mainWindow.View(),
	}

	if u.selectedColumn == 0 {
		views[0] = selectedColumnStyle.Width(int(float32(u.width) * 0.25)).Render(views[0])
		views[1] = columnStyle.Width(int(float32(u.width) * 0.75)).Render(views[1])
	} else {
		views[0] = columnStyle.Width(int(float32(u.width) * 0.25)).Render(views[0])
		views[1] = selectedColumnStyle.Width(int(float32(u.width) * 0.75)).Render(views[1])
	}

	return tea.NewView(lipgloss.JoinHorizontal(
		lipgloss.Top,
		views...,
	))
}
