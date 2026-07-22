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
	return &App{
		teaProgram: tea.NewProgram(ui.NewUI()),
	}
}

func (a App) Run() error {
	_, err := a.teaProgram.Run()
	if err != nil {
		return fmt.Errorf("app bubbletea error: %w", err)
	}
	return nil
}
