package charts

import "strings"

// baseAbbrev shortens the base measurement name (before the _mean/_se/_worst
// suffix) so it fits as a chart axis tick label.
var baseAbbrev = map[string]string{
	"radius":            "radi",
	"texture":           "text",
	"perimeter":         "peri",
	"area":              "area",
	"smoothness":        "smoo",
	"compactness":       "comp",
	"concavity":         "conc",
	"concave_points":    "conpoi",
	"symmetry":          "symm",
	"fractal_dimension": "fracdim",
}

var statAbbrev = map[string]string{
	"mean":  "M",
	"se":    "S",
	"worst": "W",
}

// abbreviateFeature turns e.g. "concave_points_worst" into "conpoi.W" so it
// fits as a chart axis tick label.
func abbreviateFeature(name string) string {
	for stat, code := range statAbbrev {
		suffix := "_" + stat
		if !strings.HasSuffix(name, suffix) {
			continue
		}
		base := strings.TrimSuffix(name, suffix)
		if abbr, ok := baseAbbrev[base]; ok {
			return abbr + "." + code
		}
		return base + "." + code
	}
	return name
}
