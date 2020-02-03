package common

import "image"

type Chip struct {
	X  int
	Y  int
	Im image.Image
}

type Detect struct {
	Bounds     image.Rectangle
	Class      int
	Chip       *Chip
	Confidence float32
}