package styles

import "charm.land/lipgloss/v2"

var (
	FormLabelStyle = lipgloss.NewStyle().
		Foreground(PrimaryColor[900])

	FormLabelFocusedStyle = lipgloss.NewStyle().
		Bold(true).
		Foreground(PrimaryColor[500])

	FormErrorStyle = lipgloss.NewStyle().
		Foreground(lipgloss.Color("#E06C75"))

	FormHintStyle = lipgloss.NewStyle().
		Foreground(PrimaryColor[800]).
		Italic(true)
)
