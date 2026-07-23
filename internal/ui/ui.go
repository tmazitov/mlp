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
	selectedColumn     int
	width              int
	height             int
}

func NewUI() *UI {

	trainingMenuTab := tabs.NewTrainingMenuTab()
	trainingProcessTab := tabs.NewTrainingProcessTab()
	predictTab := tabs.NewPredictTab()

	// allTabs is every tab MainWindow can display, including ones hidden
	// from the sidebar. menuTabs is only the subset shown in the sidebar —
	// trainingProcessTab is reachable solely by submitting the training form.
	allTabs := []tabs.Tab{trainingMenuTab, trainingProcessTab, predictTab}
	menuTabs := []tabs.Tab{trainingMenuTab, predictTab}

	var ui *UI = &UI{
		mainWindow:         views.NewMainWindow(allTabs),
		trainingProcessTab: trainingProcessTab,
		selectedColumn:     0,
	}

	ui.menuSideBar = views.NewMenuSideBar(menuTabs, ui.ActivateTab)

	return ui
}

// Init returns an initial commands for the app
func (u UI) Init() tea.Cmd {
	return nil // no init command
}

func (u UI) ActivateTab(tabName string) {
	u.mainWindow.SetCurrentTab(tabName)
	u.selectedColumn = 1
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
	case tabs.SwitchTabMsg:
		u.mainWindow.SetCurrentTab(message.TabName)
		u.selectedColumn = 1
	case tabs.AddLogMsg:
		u.trainingProcessTab.AddLog(message.Message)
	case tabs.UpdateProgressStatusMsg:
		u.trainingProcessTab.UpdateProgressStatus(message.Value)
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
