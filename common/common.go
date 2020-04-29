package common

import (
	"bytes"
	"golang.org/x/image/tiff"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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

type YoloLabel struct {
	Class CID
	X     float64
	Y     float64
	W     float64
	H     float64
}

func LoadJpeg(imagefile string) (image.Image, error) {
	file, err := os.Open(imagefile)
	if err != nil {
		return nil, err
	}

	ext := strings.ToLower(filepath.Ext(imagefile))

	var im image.Image
	if ext == "tiff" {
		im, err = tiff.Decode(file)
		if err != nil {
			return nil, err
		}
	} else {
		im, _, err = image.Decode(file)
		if err != nil {
			return nil, err
		}
	}

	buf := new(bytes.Buffer)
	if err := jpeg.Encode(buf, im, nil); err != nil {
		return nil, err
	}

	ret, err := jpeg.Decode(buf)
	if err != nil {
		return nil, err
	}

	return ret, nil
}
