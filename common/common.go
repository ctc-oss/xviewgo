package common

import (
	"image"
	"strconv"
)

type TID int
type DID int
type CID int

type Chip struct {
	X  int
	Y  int
	Im image.Image
}

type Truth struct {
	Id     TID
	Bounds image.Rectangle
	Class  CID
}

type Detect struct {
	Id         DID
	Bounds     image.Rectangle
	Class      CID
	Chip       *Chip
	Confidence float32
}

type Match struct {
	T   Truth
	D   Detect
	IoU float32
}

// (xmin,ymin,xmax,ymax)
func SplitToRect(a []string) image.Rectangle {
	mx, _ := strconv.Atoi(a[0])
	my, _ := strconv.Atoi(a[1])
	Mx, _ := strconv.Atoi(a[2])
	My, _ := strconv.Atoi(a[3])

	return image.Rectangle{
		Min: image.Point{X: mx, Y: my},
		Max: image.Point{X: Mx, Y: My},
	}
}
