package tabs

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"mlp/internal/ui/styles"

	tea "charm.land/bubbletea/v2"
)

// SplitConfig is a validated submission of the split form.
type SplitConfig struct {
	trainRatio float64
	shuffle    bool
	// seed is nil when the field was left blank, meaning "shuffle
	// differently every run".
	seed *uint64
}

var allowedShuffleAnswers = []string{"yes", "no"}

const (
	fieldRatio = iota
	fieldShuffle
	fieldSeed
	splitFieldCount
)

type SplitForm struct {
	fields    []formField
	focus     int
	errorMsg  string
	submitted bool
}

func NewSplitForm() *SplitForm {
	fields := make([]formField, splitFieldCount)
	fields[fieldRatio] = newFormField("Ratio (training/validation)", "80/20")
	fields[fieldShuffle] = newFormField(
		fmt.Sprintf("Shuffle rows first (%s)", strings.Join(allowedShuffleAnswers, ", ")), "yes")
	fields[fieldSeed] = newFormField("Seed (blank = different every run)", "42")

	// Prefill the two required fields: a placeholder is only a hint, so an
	// untouched field reads as empty and fails validation. The seed is left
	// blank on purpose — that is what asks for a fresh split each run.
	fields[fieldRatio].input.SetValue("80/20")
	fields[fieldShuffle].input.SetValue("yes")

	form := &SplitForm{fields: fields}
	form.fields[form.focus].input.Focus()

	return form
}

func (f *SplitForm) focusField(i int) tea.Cmd {
	for idx := range f.fields {
		f.fields[idx].input.Blur()
	}
	f.focus = i
	return f.fields[f.focus].input.Focus()
}

func (f *SplitForm) Update(message tea.KeyMsg) tea.Cmd {
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

// TakeSubmission reports whether the form was just submitted, consuming the
// flag so it only fires once per submit.
func (f *SplitForm) TakeSubmission() (SplitConfig, bool) {
	if !f.submitted {
		return SplitConfig{}, false
	}
	f.submitted = false
	cfg, _ := f.Value()
	return cfg, true
}

// SetError shows a message under the form, for failures that only surface
// after submission (reading the dataset, writing the output files).
func (f *SplitForm) SetError(message string) {
	f.errorMsg = message
}

func (f *SplitForm) Value() (SplitConfig, error) {
	var cfg SplitConfig

	ratio, err := parseRatio(f.fields[fieldRatio].input.Value())
	if err != nil {
		return cfg, fmt.Errorf("ratio: %w", err)
	}
	cfg.trainRatio = ratio

	shuffle := strings.TrimSpace(f.fields[fieldShuffle].input.Value())
	if !slices.Contains(allowedShuffleAnswers, shuffle) {
		return cfg, fmt.Errorf("shuffle: must be one of %s", strings.Join(allowedShuffleAnswers, ", "))
	}
	cfg.shuffle = shuffle == "yes"

	seed, err := parseSeed(f.fields[fieldSeed].input.Value())
	if err != nil {
		return cfg, fmt.Errorf("seed: %w", err)
	}
	cfg.seed = seed

	return cfg, nil
}

// parseSeed treats a blank field as "no seed", which is the only way to ask
// for a split that differs between runs.
func parseSeed(raw string) (*uint64, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}

	seed, err := strconv.ParseUint(raw, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("must be a non-negative whole number, or blank")
	}

	return &seed, nil
}

// parseRatio turns "80/20" into the training share, 0.8. The two parts must
// add up to 100 so the split is unambiguous — "80/30" is a typo, not a
// request to drop or duplicate rows.
func parseRatio(raw string) (float64, error) {
	parts := strings.Split(strings.TrimSpace(raw), "/")
	if len(parts) != 2 {
		return 0, fmt.Errorf("expected training/validation, e.g. 80/20")
	}

	train, err := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	if err != nil {
		return 0, fmt.Errorf("invalid training share %q", strings.TrimSpace(parts[0]))
	}
	validation, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err != nil {
		return 0, fmt.Errorf("invalid validation share %q", strings.TrimSpace(parts[1]))
	}

	if train <= 0 || validation <= 0 {
		return 0, fmt.Errorf("both shares must be greater than 0")
	}
	if train+validation != 100 {
		return 0, fmt.Errorf("shares must add up to 100, got %g", train+validation)
	}

	return train / 100, nil
}

func (f *SplitForm) View() string {
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
