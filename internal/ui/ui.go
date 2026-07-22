package ui

import (
	"mlp/internal/ui/styles"
	"mlp/internal/ui/views"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

type UI struct {
	columns        []Column
	proportions    []float32
	selectedColumn int
	width          int
	height         int
}

type Column interface {
	View() string
	Update(message tea.KeyMsg) tea.Cmd
}

func NewUI() *UI {
	return &UI{
		columns: []Column{
			views.NewMenuSideBar(tabs),
			views.NewMainView(),
		},
		proportions: []float32{
			0.25,
			0.75,
		},
		selectedColumn: 0,
	}
}

// Init returns an initial commands for the app
func (u UI) Init() tea.Cmd {
	return nil // no init command
}

// Update handlers incoming messages and updates the model
func (u UI) Update(message tea.Msg) (tea.Model, tea.Cmd) {

	var cmd tea.Cmd

	switch message := message.(type) {
	case tea.WindowSizeMsg:
		u.width = message.Width
		u.height = message.Height
	case tea.KeyMsg:
		switch message.String() {

		case "q", "ctrl+c":
			return u, tea.Quit

		case "right":
			u.selectedColumn = min(len(u.columns)-1, u.selectedColumn+1)

		case "left":
			u.selectedColumn = max(0, u.selectedColumn-1)

		default:
			currentColumn := u.columns[u.selectedColumn]
			cmd = currentColumn.Update(message)
		}
	}

	return u, cmd
}

// View renders the current state
func (u UI) View() tea.View {

	columnViews := make([]string, 0, len(u.columns))

	columnStyle := styles.BoxStyle.
		Height(u.height)

	for i, column := range u.columns {
		view := columnStyle.Width(int(float32(u.width) * u.proportions[i])).Render(column.View())

		columnViews = append(columnViews, view)
	}

	return tea.NewView(lipgloss.JoinHorizontal(
		lipgloss.Top,
		columnViews...,
	))
}
