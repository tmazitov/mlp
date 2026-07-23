package tabs

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"mlp/internal/ui/styles"

	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"
)

type TrainingConfig struct {
	layers       []uint16
	epochs       uint16
	lossFunc     string
	batchSize    string
	learningRate float64
}

var allowedLossFunctions = []string{"mse", "cross-entropy"}

const (
	fieldLayers = iota
	fieldEpochs
	fieldLossFunc
	fieldBatchSize
	fieldLearningRate
	fieldCount
)

type formField struct {
	label string
	input textinput.Model
}

func newFormField(label, placeholder string) formField {
	input := textinput.New()
	input.Placeholder = placeholder
	input.SetVirtualCursor(false)
	input.CharLimit = 156
	input.SetWidth(24)

	return formField{label: label, input: input}
}

type TrainingForm struct {
	fields   []formField
	focus    int
	errorMsg string
}

func NewTrainingForm() *TrainingForm {
	fields := make([]formField, fieldCount)
	fields[fieldLayers] = newFormField("Layers (comma-separated)", "784,128,64,10")
	fields[fieldEpochs] = newFormField("Epochs", "10")
	fields[fieldLossFunc] = newFormField(fmt.Sprintf("Loss function (%s)", strings.Join(allowedLossFunctions, ", ")), "mse")
	fields[fieldBatchSize] = newFormField("Batch size", "32")
	fields[fieldLearningRate] = newFormField("Learning rate", "0.01")

	form := &TrainingForm{fields: fields}
	form.fields[form.focus].input.Focus()

	return form
}

// focusField blurs every field and focuses the one at index i.
func (f *TrainingForm) focusField(i int) tea.Cmd {
	for idx := range f.fields {
		f.fields[idx].input.Blur()
	}
	f.focus = i
	return f.fields[f.focus].input.Focus()
}

func (f *TrainingForm) Update(message tea.KeyMsg) tea.Cmd {
	switch message.String() {
	case "tab", "down":
		return f.focusField((f.focus + 1) % len(f.fields))
	case "shift+tab", "up":
		return f.focusField((f.focus - 1 + len(f.fields)) % len(f.fields))
	case "enter":
		if f.focus != len(f.fields)-1 {
			return f.focusField(f.focus + 1)
		}
		if _, err := f.Value(); err != nil {
			f.errorMsg = err.Error()
			return nil
		}
		f.errorMsg = ""
		return SwitchTabCmd("training_process")
	}

	var cmd tea.Cmd
	f.fields[f.focus].input, cmd = f.fields[f.focus].input.Update(message)
	return cmd
}

// Value parses and validates the form fields into a TrainingConfig.
func (f *TrainingForm) Value() (TrainingConfig, error) {
	var cfg TrainingConfig

	layers, err := parseLayers(f.fields[fieldLayers].input.Value())
	if err != nil {
		return cfg, fmt.Errorf("layers: %w", err)
	}
	cfg.layers = layers

	epochs, err := strconv.ParseUint(strings.TrimSpace(f.fields[fieldEpochs].input.Value()), 10, 16)
	if err != nil {
		return cfg, fmt.Errorf("epochs: %w", err)
	}
	cfg.epochs = uint16(epochs)

	lossFunc := strings.TrimSpace(f.fields[fieldLossFunc].input.Value())
	if !slices.Contains(allowedLossFunctions, lossFunc) {
		return cfg, fmt.Errorf("loss function: must be one of %s", strings.Join(allowedLossFunctions, ", "))
	}
	cfg.lossFunc = lossFunc

	batchSize := strings.TrimSpace(f.fields[fieldBatchSize].input.Value())
	if batchSize == "" {
		return cfg, fmt.Errorf("batch size: required")
	}
	cfg.batchSize = batchSize

	learningRate, err := strconv.ParseFloat(strings.TrimSpace(f.fields[fieldLearningRate].input.Value()), 64)
	if err != nil {
		return cfg, fmt.Errorf("learning rate: %w", err)
	}
	cfg.learningRate = learningRate

	return cfg, nil
}

func parseLayers(raw string) ([]uint16, error) {
	layers := make([]uint16, 0)
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.ParseUint(part, 10, 16)
		if err != nil {
			return nil, fmt.Errorf("invalid layer size %q", part)
		}
		layers = append(layers, uint16(v))
	}
	if len(layers) == 0 {
		return nil, fmt.Errorf("at least one layer required")
	}
	return layers, nil
}

func (f *TrainingForm) View() string {
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

	b.WriteString(styles.FormHintStyle.Render("tab/shift+tab: move focus • enter: next field / submit"))

	return b.String()
}
