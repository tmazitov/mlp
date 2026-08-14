package tabs

import (
	"fmt"
	"strconv"
	"strings"

	"mlp/internal/analytics"
	"mlp/internal/analytics/log"
	"mlp/internal/network"
	"mlp/internal/network/activation"
	"mlp/internal/ui/styles"

	tea "charm.land/bubbletea/v2"
)

type TrainingMenuTab struct {
	form *TrainingForm
}

func NewTrainingMenuTab() *TrainingMenuTab {
	return &TrainingMenuTab{
		form: NewTrainingForm(),
	}
}

func (t TrainingMenuTab) Name() string  { return "training_menu" }
func (t TrainingMenuTab) Title() string { return "Training" }

func (t *TrainingMenuTab) Update(message tea.KeyMsg) tea.Cmd {
	cmd := t.form.Update(message)

	cfg, ok := t.form.TakeSubmission()
	if !ok {
		return cmd
	}

	model, logs, err := buildModel(cfg)
	if err != nil {
		t.form.errorMsg = err.Error()
		return cmd
	}

	dataset, err := analytics.Load(datasetCSVPath)
	if err != nil {
		t.form.errorMsg = fmt.Sprintf("load dataset: %v", err)
		return cmd
	}

	fieldsToTrain := []string{
		"area_worst",
		"concave_points_worst",
		"concavity_mean",
		"texture_worst",
		"smoothness_worst",
		"symmetry_worst",
		"fractal_dimension_worst",
		"compactness_worst",
	}

	dataset = dataset.ExtractFields(fieldsToTrain...)

	return tea.Batch(cmd, SwitchTabCmd("training_process"), StartTrainingCmd(model, dataset, logs, int(cfg.epochs)))
}

// buildModel turns a validated TrainingConfig into a ready-to-run MLP. The
// input and output layers are fixed by the app, not the user: an 8-neuron
// sigmoid input layer (matching the 8 fields selected in fieldsToTrain) and
// a 2-neuron softmax output layer (paired with cross-entropy, per
// network.Train's softmax+cross-entropy shortcut). cfg.layers only controls
// the sigmoid hidden layers in between.
func buildModel(cfg TrainingConfig) (*network.MLP, chan log.TrainingStat, error) {
	batchSize, err := strconv.Atoi(cfg.batchSize)
	if err != nil {
		return nil, nil, fmt.Errorf("batch size: %w", err)
	}

	logs := make(chan log.TrainingStat)

	mlpConfig := network.MLPConfig{
		Epochs:    int(cfg.epochs),
		BatchSize: batchSize,
		Mode:      network.TrainingMode,
		LogsChan:  logs,
	}
	switch cfg.lossFunc {
	case "mse":
		mlpConfig.LossFunc = network.MSELossFunc
	case "cross-entropy":
		mlpConfig.LossFunc = network.CrossEntropyLossFunc
	}

	model := network.NewMLP(mlpConfig)

	sigmoid := activation.SigmoidFunc{}
	softmax := activation.SoftMaxFunc{}

	// Inner layer (hardcoded and based on parameters)
	if err := model.AddLayer(8, sigmoid); err != nil {
		return nil, nil, err
	}

	// Hidden layers
	for _, size := range cfg.layers {
		if err := model.AddLayer(uint(size), sigmoid); err != nil {
			return nil, nil, err
		}
	}

	// Outer layer
	if err := model.AddLayer(2, softmax); err != nil {
		return nil, nil, err
	}

	return model, logs, nil
}

func (t TrainingMenuTab) View() string {
	var contentBuilder strings.Builder

	contentBuilder.WriteString(styles.TabTitleStyle.Render(t.Title()))
	contentBuilder.WriteRune('\n')
	contentBuilder.WriteString(t.form.View())

	return contentBuilder.String()
}
