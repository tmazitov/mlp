package tabs

import (
	"fmt"
	"strings"

	"mlp/internal/analytics"
	"mlp/internal/ui/styles"

	tea "charm.land/bubbletea/v2"
)

// The split phase writes these two files, and the training phase reads
// them. Keeping the names fixed is what ties the two phases together.
const (
	trainingCSVPath   = "data_training.csv"
	validationCSVPath = "data_validation.csv"
)

type SplitTab struct {
	form   *SplitForm
	result *splitResult
}

// splitResult is what the last successful split produced, kept so View can
// report it after the fact.
type splitResult struct {
	trainRows  int
	valRows    int
	trainClass map[string]int
	valClass   map[string]int
	shuffled   bool
	seed       *uint64
}

func NewSplitTab() *SplitTab {
	return &SplitTab{form: NewSplitForm()}
}

func (t *SplitTab) Name() string  { return "split" }
func (t *SplitTab) Title() string { return "Split" }

func (t *SplitTab) Update(message tea.KeyMsg) tea.Cmd {
	cmd := t.form.Update(message)

	cfg, ok := t.form.TakeSubmission()
	if !ok {
		return cmd
	}

	if err := t.split(cfg); err != nil {
		t.result = nil
		t.form.SetError(err.Error())
	}

	return cmd
}

// split reads the full dataset, divides it and writes both halves back out
// as CSV in the same shape Load reads, so the training phase can pick them
// up without any special casing.
func (t *SplitTab) split(cfg SplitConfig) error {
	dataset, err := analytics.Load(datasetCSVPath)
	if err != nil {
		return fmt.Errorf("load %s: %w", datasetCSVPath, err)
	}

	trainSet, valSet := dataset.Split(cfg.trainRatio, cfg.shuffle, cfg.seed)

	if err := trainSet.WriteCSV(trainingCSVPath); err != nil {
		return err
	}
	if err := valSet.WriteCSV(validationCSVPath); err != nil {
		return err
	}

	t.result = &splitResult{
		trainRows:  len(trainSet.Rows),
		valRows:    len(valSet.Rows),
		trainClass: trainSet.ClassCounts(),
		valClass:   valSet.ClassCounts(),
		shuffled:   cfg.shuffle,
		seed:       cfg.seed,
	}

	return nil
}

func (t *SplitTab) View() string {
	var b strings.Builder

	b.WriteString(styles.TabTitleStyle.Render(t.Title()))
	b.WriteRune('\n')
	b.WriteString(styles.DescriptionStyle.Render(
		fmt.Sprintf("Divides %s into a training and a validation file.", datasetCSVPath)))
	b.WriteString("\n\n")

	b.WriteString(t.form.View())

	if t.result != nil {
		b.WriteString("\n\n")
		b.WriteString(styles.FormLabelFocusedStyle.Render("Done"))
		b.WriteRune('\n')

		order := "kept the original row order"
		if t.result.shuffled {
			order = "rows shuffled before splitting"
			if t.result.seed != nil {
				order += fmt.Sprintf(" (seed %d — rerun with it to get these exact files again)", *t.result.seed)
			} else {
				order += " (no seed — a rerun will deal a different split)"
			}
		}
		b.WriteString(styles.DescriptionStyle.Render(order))
		b.WriteString("\n\n")

		b.WriteString(styles.DescriptionStyle.Render(fmt.Sprintf(
			"%s — %d rows (B %d / M %d)",
			fileLink(trainingCSVPath), t.result.trainRows,
			t.result.trainClass["B"], t.result.trainClass["M"])))
		b.WriteRune('\n')
		b.WriteString(styles.DescriptionStyle.Render(fmt.Sprintf(
			"%s — %d rows (B %d / M %d)",
			fileLink(validationCSVPath), t.result.valRows,
			t.result.valClass["B"], t.result.valClass["M"])))
	}

	return b.String()
}
