package styles

import "charm.land/lipgloss/v2"

var (
	TabTitleStyle = lipgloss.NewStyle().
		Bold(true).
		Foreground(PrimaryColor[500]).
		Padding(0, 1)
)
