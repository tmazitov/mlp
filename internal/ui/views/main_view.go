package views

import (
	"strings"

	tea "charm.land/bubbletea/v2"
)

type MainView struct {
}

func NewMainView() *MainView {
	return &MainView{}
}

func (v MainView) Update(message tea.KeyMsg) tea.Cmd {
	return nil
}

func (v MainView) View() string {
	var contentBuilder strings.Builder

	contentBuilder.WriteString("                                        \n\n\n\n\n\n\n\n\n")

	return contentBuilder.String()
}
