package tabs

import (
	"fmt"
	"strconv"
	"strings"

	"mlp/internal/analytics/log"
	"mlp/internal/network"
	"mlp/internal/network/activation"
	"mlp/internal/network/neuron"
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

	data, err := loadTrainingData()
	if err != nil {
		t.form.errorMsg = err.Error()
		return cmd
	}

	return tea.Batch(cmd, SwitchTabCmd("training_process"),
		StartTrainingCmd(model, data.train, data.val, logs, int(cfg.epochs), fieldsToTrain, data.scaler))
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
		LogsChan:  logs,
	}
	switch cfg.lossFunc {
	case "cross-entropy":
		mlpConfig.LossFunc = network.CrossEntropyLossFunc
	}

	model := network.NewMLP(mlpConfig)

	optimizer := neuron.SGDOptimizer
	switch cfg.optimizer {
	case "nesterov":
		optimizer = neuron.NesterovOptimizer
	case "rmsprop":
		optimizer = neuron.RMSPropOptimizer
	}

	sigmoid := activation.SigmoidFunc{}
	softmax := activation.SoftMaxFunc{}

	// Combine input, hidden and output layers
	layers := []uint{8}
	layers = append(layers, cfg.layers...)
	layers = append(layers, 2)

	// Setup all hidden layers
	for i := 1; i <= len(layers)-2; i++ {

		neuronParams := neuron.NeuronParams{
			InitType:     neuron.XavierInitFunc,
			NIn:          layers[i-1],
			NOut:         layers[i],
			LearningRate: cfg.learningRate,
			Optimizer:    optimizer,
		}

		size := layers[i]

		if err := model.AddLayer(size, sigmoid, neuronParams); err != nil {
			return nil, nil, err
		}
	}

	// Outer layer
	if err := model.AddLayer(2, softmax, neuron.NeuronParams{
		InitType:     neuron.HeInitFunc,
		NIn:          layers[len(layers)-2],
		NOut:         layers[len(layers)-1],
		LearningRate: cfg.learningRate,
		Optimizer:    optimizer,
	}); err != nil {
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
