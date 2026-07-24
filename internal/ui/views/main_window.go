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

// SetSize tells every tab that wants to know (i.e. implements tabs.Sizable)
// how much content area it will actually be rendered into, so it can size
// internal scrollable components instead of assuming a fixed terminal size.
func (w *MainWindow) SetSize(width, height int) {
	for _, tab := range w.tabs {
		if sizable, ok := tab.(tabs.Sizable); ok {
			sizable.SetSize(width, height)
		}
	}
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
