package tabs

import (
	"strings"
	"time"

	"mlp/internal/ui/styles"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

// The bell is drawn in the palette's brightest tone and the tentacles a few
// steps darker, so the mascot reads as having depth rather than as flat text.
var (
	mascotBellStyle     = lipgloss.NewStyle().Bold(true).Foreground(styles.PrimaryColor[500])
	mascotTentacleStyle = lipgloss.NewStyle().Foreground(styles.PrimaryColor[700])
)

// MascotTickMsg advances the mascot animation by one frame. It is emitted on
// a timer by MascotTickCmd, and — like every other state change here — is
// handled inside the Update loop, so the animation never races with training.
type MascotTickMsg struct{}

// mascotFrameDuration is how long a single animation frame stays on screen.
// Four frames make up one breath, so the jellyfish breathes roughly once per
// second.
const mascotFrameDuration = 250 * time.Millisecond

// MascotTickCmd schedules the next animation frame. Re-issue it from Update
// on every MascotTickMsg for as long as the animation should keep running;
// stop returning it and the animation stops on its own.
func MascotTickCmd() tea.Cmd {
	return tea.Tick(mascotFrameDuration, func(time.Time) tea.Msg {
		return MascotTickMsg{}
	})
}

// mascotBellLines is how many of a frame's top lines belong to the bell (the
// dome), which is drawn brighter than the tentacles below it.
const mascotBellLines = 4

// mascotRelaxed, mascotMid and mascotContracted are the three body shapes of
// one breath: the bell relaxes into a tall, narrow dome and then squeezes
// into a flat, wide one — the same way a real jellyfish swims.
const (
	mascotRelaxed = `
    .-~~~-.
   /       \
  |  o   o  |
   \  ___  /
    ) | | (
   (  | |  )
     '   '`

	mascotMid = `
   .-~~~~-.
  /        \
 |  o    o  |
  \   ___  /
   (  | |  )
    ) | | (
     '   '`

	mascotContracted = `
  .-~~~~~-.
 /         \
|  o     o  |
 \   ___   /
  )  | |  (
 (   | |   )
    '   '`
)

// mascotFrame is one animation step: a body shape plus how far down it floats.
// Squeezing the bell pushes the jellyfish up, relaxing it lets it sink back —
// so shape and offset together read as breathing, not just as a blinking
// drawing.
type mascotFrame struct {
	body   string
	offset int
}

var mascotFrames = []mascotFrame{
	{body: mascotRelaxed, offset: 1},
	{body: mascotMid, offset: 1},
	{body: mascotContracted, offset: 0},
	{body: mascotMid, offset: 0},
}

// mascotHeight keeps every rendered frame the same number of lines, so the
// content below the mascot never jumps as it bobs.
const mascotHeight = 8

// Mascot is the breathing jellyfish shown while the model trains. It holds
// nothing but the current frame index; call Tick to advance it.
type Mascot struct {
	frame int
}

// Tick advances the mascot to its next animation frame.
func (m *Mascot) Tick() {
	m.frame = (m.frame + 1) % len(mascotFrames)
}

// View renders the current frame, padded to a fixed width and height so the
// surrounding layout stays still while the jellyfish moves inside it.
func (m Mascot) View() string {
	current := mascotFrames[m.frame]
	bodyLines := strings.Split(strings.Trim(current.body, "\n"), "\n")

	width := 0
	for _, line := range bodyLines {
		width = max(width, len(line))
	}

	lines := make([]string, 0, mascotHeight)
	for range current.offset {
		lines = append(lines, strings.Repeat(" ", width))
	}

	for index, line := range bodyLines {
		padded := line + strings.Repeat(" ", width-len(line))

		style := mascotTentacleStyle
		if index < mascotBellLines {
			style = mascotBellStyle
		}
		lines = append(lines, style.Render(padded))
	}

	for len(lines) < mascotHeight {
		lines = append(lines, strings.Repeat(" ", width))
	}

	return strings.Join(lines, "\n")
}
