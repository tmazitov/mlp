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
