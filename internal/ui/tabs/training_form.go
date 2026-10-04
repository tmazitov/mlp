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
	layers       []uint
	epochs       uint16
	lossFunc     string
	batchSize    string
	learningRate float64
}

var allowedLossFunctions = []string{"cross-entropy"}

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
	fields    []formField
	focus     int
	errorMsg  string
	submitted bool
}

func NewTrainingForm() *TrainingForm {
	fields := make([]formField, fieldCount)
	fields[fieldLayers] = newFormField("Layers (comma-separated)", "8, 8, 8")
	fields[fieldEpochs] = newFormField("Epochs", "3000")
	fields[fieldLossFunc] = newFormField(
		fmt.Sprintf("Loss function (%s)", strings.Join(allowedLossFunctions, ", ")), "cross-entropy")
	fields[fieldBatchSize] = newFormField("Batch size", "16")
	fields[fieldLearningRate] = newFormField("Learning rate", "0.05")

	// Prefill rather than rely on the placeholders: a placeholder is only a
	// hint, so an untouched field still reads as empty and fails validation.
	// The defaults give two hidden layers, as the subject requires, and
	// hyperparameters that are known to converge on this dataset.
	for i, value := range map[int]string{
		fieldLayers:       "8, 8",
		fieldEpochs:       "3000",
		fieldLossFunc:     "cross-entropy",
		fieldBatchSize:    "16",
		fieldLearningRate: "0.05",
	} {
		fields[i].input.SetValue(value)
	}

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
		f.submitted = true
		return nil
	}

	var cmd tea.Cmd
	f.fields[f.focus].input, cmd = f.fields[f.focus].input.Update(message)
	return cmd
}

// TakeSubmission reports whether the form was just submitted (all fields
// valid), consuming the flag so it only fires once per submit. The config
// was already validated inside Update, so the re-parse here cannot fail.
func (f *TrainingForm) TakeSubmission() (TrainingConfig, bool) {
	if !f.submitted {
		return TrainingConfig{}, false
	}
	f.submitted = false
	cfg, _ := f.Value()
	return cfg, true
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

func parseLayers(raw string) ([]uint, error) {
	layers := make([]uint, 0)
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.ParseUint(part, 10, 16)
		if err != nil {
			return nil, fmt.Errorf("invalid layer size %q", part)
		}
		layers = append(layers, uint(v))
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
