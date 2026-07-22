package views

import (
	"mlp/internal/ui/styles"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Black).
			Background(styles.PrimaryColor[500]).
			Padding(0, 1)

	itemStyle = lipgloss.NewStyle().
			Foreground(styles.PrimaryColor[900]).
			PaddingLeft(1)

	selectedStyle = lipgloss.NewStyle().
			PaddingLeft(1).
			Foreground(styles.PrimaryColor[400]).
			Bold(true)

	hoveredStyle = lipgloss.NewStyle().
			PaddingLeft(1).
			Foreground(styles.PrimaryColor[700]).
			Bold(true)

	creditsStyle = lipgloss.NewStyle().
			Foreground(styles.PrimaryColor[900])
)

type MenuSideBar struct {
	tabs        []string
	hoveredTab  int
	selectedTab int
}

func NewMenuSideBar(tabs []string) *MenuSideBar {
	return &MenuSideBar{
		tabs:        tabs,
		hoveredTab:  0,
		selectedTab: -1,
	}
}

func (m MenuSideBar) View() string {
	var contentBuilder strings.Builder

	title := titleStyle.Render("Multilayer Perceptron (MLP)")
	contentBuilder.WriteString(title)
	contentBuilder.WriteString("\n\n\n")

	for i, item := range m.tabs {
		contentBuilder.WriteRune('\n')
		if i == m.hoveredTab {
			contentBuilder.WriteString(selectedStyle.Render("> "))
		} else {
			contentBuilder.WriteString(itemStyle.Render("  "))
		}

		var textStyle lipgloss.Style
		switch i {
		case m.selectedTab:
			textStyle = selectedStyle
		case m.hoveredTab:
			textStyle = hoveredStyle
		default:
			textStyle = itemStyle
		}
		contentBuilder.WriteString(textStyle.Render(item))
		contentBuilder.WriteRune('\n')
	}

	credits := creditsStyle.Render("tmazitov * 2026")
	contentBuilder.WriteString("\n\n")
	contentBuilder.WriteString(credits)

	return contentBuilder.String()
}

func (m *MenuSideBar) Update(message tea.KeyMsg) tea.Cmd {

	switch message.String() {

	case "up", "k":
		// Increment the counter
		m.hoveredTab--
		if m.hoveredTab < 0 {
			m.hoveredTab = len(m.tabs) - 1
		}
	case "down", "j":
		// Decrement the counter
		m.hoveredTab++
		if m.hoveredTab > len(m.tabs)-1 {
			m.hoveredTab = 0
		}
	case "enter", "space":
		m.selectedTab = m.hoveredTab
	}

	return nil
}
