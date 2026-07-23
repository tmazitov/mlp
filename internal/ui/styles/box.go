package styles

import "charm.land/lipgloss/v2"

var (
	BoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(PrimaryColor[800]).
			Padding(1, 2, 0, 2)

	SelectedBoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(PrimaryColor[600]).
			Padding(1, 2, 0, 2)
)
