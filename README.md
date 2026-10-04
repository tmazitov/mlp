# Multilayer Perceptron

A multilayer perceptron written from scratch in Go, trained to predict
whether a breast-mass cell nucleus is malignant or benign from the
Wisconsin diagnostic dataset (`data.csv`, 569 rows, 30 features).

Everything in the network — feedforward, backpropagation, gradient descent,
the activation functions and the loss — is implemented here. The only
third-party library is `gonum/plot`, used to draw the charts.

## Build

```sh
make          # builds ./mlp and ./analyze
make re       # rebuild from scratch
make fclean   # remove the binaries
```

## Run

```sh
./mlp         # the application, all phases
./analyze     # regenerate the dataset exploration charts
```

The app is a terminal UI. Move between panes with `←` / `→`, pick a tab with
`↑` / `↓` and `enter`, move between form fields with `tab` / `shift+tab`,
submit with `enter` on the last field, and quit with `q`.

## The three phases

The subject asks for dataset separation, training and prediction. Each is a
tab, run in this order.

### 1. Split

Divides `data.csv` into `data_training.csv` and `data_validation.csv`.

| Field | Meaning |
| --- | --- |
| Ratio | `80/20` — training share first, the two must add up to 100 |
| Shuffle rows first | `yes` randomises before dividing, `no` keeps file order |
| Seed | a number makes the shuffle reproducible; blank deals a new split each run |

Both files keep the dataset's original 32-column layout, so they can be fed
back into any phase.

### 2. Training

Reads the two files from the split phase, trains, and writes `model.json`.

| Field | Default | Meaning |
| --- | --- | --- |
| Layers | `8, 8` | neurons per hidden layer; the input and output layers are fixed |
| Epochs | `3000` | passes over the training set |
| Loss function | `cross-entropy` | the only one implemented, and what the backward pass optimises |
| Batch size | `16` | rows per weight update |
| Learning rate | `0.05` | step size |
| Optimizer | `sgd` | how the gradient becomes a step |
| Early stop patience | `0` | stop after this many epochs without a better validation loss; `0` runs every epoch |

### Optimizers

| Name | Update | Notes |
| --- | --- | --- |
| `sgd` | `w -= lr * g` | the plain step |
| `nesterov` | momentum, `v = 0.9v + g`, `w -= lr * (g + 0.9v)` | accelerates along a consistent direction and damps zig-zagging; converges noticeably faster early on |
| `rmsprop` | `s = 0.9s + 0.1g²`, `w -= lr * g / (sqrt(s) + 1e-8)` | scales each parameter's step by its own recent gradient size |

RMSProp normalises the step by gradient magnitude rather than scaling by
it, so it wants a much smaller learning rate than the others — around
`0.001`. At `0.05` it reaches a good validation loss within tens of epochs
and then degrades.

### What training reports

Every epoch is scored on both sets and logged as loss, accuracy, and —
for validation — precision, recall and F1. Accuracy alone hides the
mistake that matters here: the dataset is about 63% benign, so a model
that answered "benign" to everything would already score 0.63 while
missing every malignant case. The confusion matrix behind these numbers
separates the two kinds of error.

| Metric | Formula | Reads as |
| --- | --- | --- |
| Accuracy | `(TP + TN) / rows` | how often the answer is right |
| Precision | `TP / (TP + FP)` | of the rows called malignant, how many were |
| Recall | `TP / (TP + FN)` | of the malignant rows, how many were caught |
| Specificity | `TN / (TN + FP)` | of the benign rows, how many were left alone |
| F1 | harmonic mean of precision and recall | one number when both matter |

Recall is the one to watch: a false negative is a malignant tumour called
benign.

On completion the run writes to `training_output/`:

- `loss_curve.png` — training and validation loss per epoch
- `accuracy_curve.png` — training and validation accuracy per epoch
- `history_<timestamp>.csv` — every metric above, both sets, one row per
  epoch, for plotting the run elsewhere or comparing two runs after the fact

### Early stopping

With a patience set, the run keeps a copy of the weights from the best
validation loss it has seen. If that loss fails to improve for `patience`
epochs in a row, training stops and those weights are put back — the
epochs since the best one were spent fitting the training set at the
expense of data the model does not learn from.

Rolling back is the part that matters, not stopping early. On a `8, 8`
nesterov run over 400 epochs:

| | Epochs run | Final validation loss |
| --- | --- | --- |
| No patience | 400 | 0.1439 |
| Patience 20 | 157, restored to 137 | 0.0171 |

Both runs passed through roughly the same best point; without the
rollback the model keeps going and arrives somewhere worse.

### Comparing configurations

The **Models** tab keeps a list of configurations and trains the selected
ones at the same time, each in its own goroutine, charting them together
once all have finished.

| Key | Does |
| --- | --- |
| `↑` / `↓` | move through the list |
| `space` | select or deselect a configuration |
| `a` | add a new one, using the training form |
| `d` | delete the highlighted one |
| `enter` | train everything selected |

The result is two charts in `training_output/`, one line per model:
`comparison_loss.png` and `comparison_accuracy.png`. They plot validation
curves only — adding each model's training curve would double the lines to
say how well a model fits rows it has already seen, which is not what is
being compared.

Each run also leaves its own `history_<name>_<timestamp>.csv`, named after
the configuration, so a row in the chart can be traced back to its numbers.

The list starts with the three optimizers at the learning rate each one
wants, which is a comparison worth seeing first.

### 3. Predict

Loads `model.json`, scores it against a labelled set (`data_validation.csv`
by default) and reports cross-entropy loss, the confusion matrix, and the
accuracy, precision, recall, specificity and F1 drawn from it.

The model file holds more than weights: it also stores the feature columns
the network was trained on and the mean/standard deviation each column was
scaled with. Prediction rebuilds the input the same way training did, so the
network sees the distribution it learned on.

## Dataset exploration

The **Dataset** tab summarises `data.csv` — class balance, per-feature
statistics by diagnosis, and strongly correlated feature pairs — and opens
four charts written to `analytics_output/` by `./analyze`:

| Chart | Shows |
| --- | --- |
| `class_balance.png` | how many rows are benign vs malignant |
| `feature_scales.png` | the orders of magnitude between features, i.e. why the inputs are standardised |
| `correlation_heatmap.png` | which features carry the same information |
| `feature_separation.png` | which features actually separate the two classes |

## Layout

```
cmd/            entry points: the app and the chart generator
internal/
  analytics/    dataset loading, splitting, standardisation, statistics
    charts/     chart rendering
    log/        per-epoch metrics and the history CSV
  network/      the MLP: layers, neurons, activations, loss, training, prediction
  ui/           terminal interface
pkg/vector/     vector arithmetic
```
