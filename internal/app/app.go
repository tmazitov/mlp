package app

import (
	"fmt"
	"mlp/internal/ui"

	tea "charm.land/bubbletea/v2"
)

type App struct {
	teaProgram *tea.Program
}

func NewApp() *App {
	uiModel := ui.NewUI()
	program := tea.NewProgram(uiModel)
	uiModel.SetProgram(program)

	return &App{
		teaProgram: program,
	}
}

func (a App) Run() error {
	_, err := a.teaProgram.Run()
	if err != nil {
		return fmt.Errorf("app bubbletea error: %w", err)
	}
	return nil
}
