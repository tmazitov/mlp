package tabs

import (
	tea "charm.land/bubbletea/v2"
)

type Tab interface {
	Update(message tea.KeyMsg) tea.Cmd
	View() string
	Title() string
	Name() string
}

// SwitchTabMsg requests that the active tab be changed to TabName. Tabs emit
// it (via a tea.Cmd) to navigate elsewhere in response to their own state,
// e.g. moving from a form to the view that shows its result.
type SwitchTabMsg struct {
	TabName string
}

func SwitchTabCmd(tabName string) tea.Cmd {
	return func() tea.Msg {
		return SwitchTabMsg{TabName: tabName}
	}
}
