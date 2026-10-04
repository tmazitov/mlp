package tabs

import (
	"strings"

	"mlp/internal/ui/styles"

	tea "charm.land/bubbletea/v2"
)

// PredictConfig is a validated submission of the predict form.
type PredictConfig struct {
	modelPath   string
	datasetPath string
}

const (
	fieldModelPath = iota
	fieldPredictDataset
	predictFieldCount
)

type PredictForm struct {
	fields    []formField
	focus     int
	errorMsg  string
	submitted bool
}

func NewPredictForm() *PredictForm {
	fields := make([]formField, predictFieldCount)
	fields[fieldModelPath] = newFormField("Model file", modelFilePath)
	fields[fieldPredictDataset] = newFormField("Dataset to predict", validationCSVPath)

	form := &PredictForm{fields: fields}
	form.fields[form.focus].input.Focus()

	return form
}

func (f *PredictForm) focusField(i int) tea.Cmd {
	for idx := range f.fields {
		f.fields[idx].input.Blur()
	}
	f.focus = i
	return f.fields[f.focus].input.Focus()
}

func (f *PredictForm) Update(message tea.KeyMsg) tea.Cmd {
	switch message.String() {
	case "tab", "down":
		return f.focusField((f.focus + 1) % len(f.fields))
	case "shift+tab", "up":
		return f.focusField((f.focus - 1 + len(f.fields)) % len(f.fields))
	case "enter":
		if f.focus != len(f.fields)-1 {
			return f.focusField(f.focus + 1)
		}
		f.errorMsg = ""
		f.submitted = true
		return nil
	}

	var cmd tea.Cmd
	f.fields[f.focus].input, cmd = f.fields[f.focus].input.Update(message)
	return cmd
}

// TakeSubmission reports whether the form was just submitted, consuming the
// flag so it only fires once per submit.
func (f *PredictForm) TakeSubmission() (PredictConfig, bool) {
	if !f.submitted {
		return PredictConfig{}, false
	}
	f.submitted = false
	return f.Value(), true
}

// SetError shows a message under the form, for failures that only surface
// once the files are actually opened.
func (f *PredictForm) SetError(message string) {
	f.errorMsg = message
}

// Value reads the two paths, falling back to the placeholders so an
// untouched form just does the obvious thing: score the validation split
// with the model the last training run wrote.
func (f *PredictForm) Value() PredictConfig {
	return PredictConfig{
		modelPath:   valueOrDefault(f.fields[fieldModelPath].input.Value(), modelFilePath),
		datasetPath: valueOrDefault(f.fields[fieldPredictDataset].input.Value(), validationCSVPath),
	}
}

func valueOrDefault(value, fallback string) string {
	if trimmed := strings.TrimSpace(value); trimmed != "" {
		return trimmed
	}
	return fallback
}

func (f *PredictForm) View() string {
	var b strings.Builder

	for i, field := range f.fields {
		labelStyle := styles.FormLabelStyle
		if i == f.focus {
			labelStyle = styles.FormLabelFocusedStyle
		}
		b.WriteString(labelStyle.Render(field.label))
		b.WriteRune('\n')
		b.WriteString(field.input.View())
		b.WriteString("\n\n")
	}

	if f.errorMsg != "" {
		b.WriteString(styles.FormErrorStyle.Render("Error: " + f.errorMsg))
		b.WriteRune('\n')
	}

	b.WriteString(styles.FormHintStyle.Render("tab/shift+tab: move focus • enter: next field / run"))

	return b.String()
}
