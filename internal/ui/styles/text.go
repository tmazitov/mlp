package styles

import "charm.land/lipgloss/v2"

var (
	DescriptionStyle = lipgloss.NewStyle().
		Foreground(PrimaryColor[900])

	LinkStyle = lipgloss.NewStyle().
		Foreground(PrimaryColor[500]).
		Underline(true)
)
