package tabs

import (
	"os/exec"
	"runtime"
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
