package tabs

import (
	"fmt"
	"strings"

	"mlp/internal/analytics"
	"mlp/internal/network"
	"mlp/internal/ui/styles"

	tea "charm.land/bubbletea/v2"
)

type PredictTab struct {
	form   *PredictForm
	result *predictResult
}

type predictResult struct {
	datasetPath string
	modelPath   string
	fields      []string
	evaluation  network.Evaluation
}

func NewPredictTab() *PredictTab {
	return &PredictTab{form: NewPredictForm()}
}

func (t *PredictTab) Name() string  { return "predict_menu" }
func (t *PredictTab) Title() string { return "Predict" }

func (t *PredictTab) Update(message tea.KeyMsg) tea.Cmd {
	cmd := t.form.Update(message)

	cfg, ok := t.form.TakeSubmission()
	if !ok {
		return cmd
	}

	if err := t.predict(cfg); err != nil {
		t.result = nil
		t.form.SetError(err.Error())
	}

	return cmd
}

// predict loads a model saved by the training phase and scores it against a
// labelled set. The dataset is passed in raw: the model carries the fields
// and scaler it was trained with, so it prepares the input itself.
func (t *PredictTab) predict(cfg PredictConfig) error {
	model, err := network.LoadModel(cfg.modelPath)
	if err != nil {
		return fmt.Errorf("%w — train a model first", err)
	}

	dataset, err := analytics.Load(cfg.datasetPath)
	if err != nil {
		return err
	}

	evaluation, err := model.Evaluate(dataset)
	if err != nil {
		return err
	}

	t.result = &predictResult{
		datasetPath: cfg.datasetPath,
		modelPath:   cfg.modelPath,
		fields:      model.Fields,
		evaluation:  evaluation,
	}

	return nil
}

func (t *PredictTab) View() string {
	var b strings.Builder

	b.WriteString(styles.TabTitleStyle.Render(t.Title()))
	b.WriteRune('\n')
	b.WriteString(styles.DescriptionStyle.Render(
		"Loads a saved model and scores it against a labelled set."))
	b.WriteString("\n\n")

	b.WriteString(t.form.View())

	if t.result != nil {
		b.WriteString("\n\n")
		b.WriteString(renderEvaluation(t.result))
	}

	return b.String()
}

func renderEvaluation(result *predictResult) string {
	var b strings.Builder
	e := result.evaluation

	b.WriteString(styles.FormLabelFocusedStyle.Render("Result"))
	b.WriteRune('\n')
	b.WriteString(styles.DescriptionStyle.Render(fmt.Sprintf(
		"%s — %d rows, %d features (%s)",
		fileLink(result.datasetPath), e.Rows, len(result.fields), strings.Join(result.fields, ", "))))
	b.WriteString("\n\n")

	b.WriteString(styles.FormLabelStyle.Render(fmt.Sprintf(
		"binary cross-entropy  %.4f", e.Loss)))
	b.WriteRune('\n')
	b.WriteString(styles.FormLabelStyle.Render(fmt.Sprintf(
		"accuracy              %.4f  (%d of %d correct)",
		e.Accuracy, e.TruePositives+e.TrueNegatives, e.Rows)))
	b.WriteString("\n\n")

	// A plain accuracy figure hides which way the model errs, and on this
	// dataset the two directions are not equally bad.
	b.WriteString(styles.FormLabelFocusedStyle.Render("Confusion matrix"))
	b.WriteRune('\n')
	b.WriteString(styles.FormLabelStyle.Render("                 predicted M   predicted B"))
	b.WriteRune('\n')
	b.WriteString(styles.FormLabelStyle.Render(fmt.Sprintf(
		"  actual M       %11d   %11d", e.TruePositives, e.FalseNegatives)))
	b.WriteRune('\n')
	b.WriteString(styles.FormLabelStyle.Render(fmt.Sprintf(
		"  actual B       %11d   %11d", e.FalsePositives, e.TrueNegatives)))
	b.WriteRune('\n')
	b.WriteString(styles.FormHintStyle.Render(fmt.Sprintf(
		"%d malignant case(s) reported benign", e.FalseNegatives)))

	return b.String()
}
