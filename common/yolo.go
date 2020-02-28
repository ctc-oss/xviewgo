package common

import "strconv"

// <object-class> <x_center> <y_center> <width> <height>
func ReadYoloLabels(csv [][]string) []YoloLabel {
	labels := make([]YoloLabel, 0)
	for _, splits := range csv {
		class, _ := strconv.Atoi(splits[0])
		x, _ := strconv.ParseFloat(splits[1], 64)
		y, _ := strconv.ParseFloat(splits[2], 64)
		w, _ := strconv.ParseFloat(splits[3], 64)
		h, _ := strconv.ParseFloat(splits[4], 64)
		labels = append(labels, YoloLabel{
			Class: CID(class),
			X:     x,
			Y:     y,
			W:     w,
			H:     h,
		})
	}
	return labels
}
