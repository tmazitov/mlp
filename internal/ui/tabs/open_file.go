package tabs

import (
	"os/exec"
	"path/filepath"
	"runtime"

	"mlp/internal/ui/styles"
)

// openInSystemViewer opens path in whatever application the OS has
// associated with it (e.g. an image viewer for a .png). It only reports
// whether the launch itself failed (missing binary, etc.) — it does not
// wait for the opened application to exit.
func openInSystemViewer(path string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", path)
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", "", path)
	default:
		cmd = exec.Command("xdg-open", path)
	}
	return cmd.Start()
}

// fileLink renders path as a clickable terminal hyperlink (OSC 8). The URL
// must be absolute — "file://" + a relative path parses the first segment as
// a hostname and no link handler can open it — so the path is resolved
// against the working directory first, and the absolute form is what gets
// displayed too, so it can be copied as-is.
func fileLink(path string) string {
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	return styles.LinkStyle.Hyperlink("file://" + path).Render(path)
}
