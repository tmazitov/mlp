package views

import (
	"mlp/internal/ui/tabs"
	"strings"

	tea "charm.land/bubbletea/v2"
)

type MainWindow struct {
	tabs           []tabs.Tab
	currentTabName string
}

func NewMainWindow(tabs []tabs.Tab) *MainWindow {
	return &MainWindow{
		tabs:           tabs,
		currentTabName: "",
	}
}

func (w *MainWindow) SetCurrentTab(tabName string) {
	w.currentTabName = tabName
}

func (w MainWindow) currentTab() tabs.Tab {
	for _, tab := range w.tabs {
		if tab.Name() == w.currentTabName {
			return tab
		}
	}

	return nil
}

func (w MainWindow) Update(message tea.KeyMsg) tea.Cmd {
	currentTab := w.currentTab()
	if currentTab == nil {
		return nil
	}

	return currentTab.Update(message)
}

func (w MainWindow) View() string {
	var contentBuilder strings.Builder

	currentTab := w.currentTab()
	if currentTab == nil {
		return contentBuilder.String()
	}

	contentBuilder.WriteString(currentTab.View())

	return contentBuilder.String()
}
