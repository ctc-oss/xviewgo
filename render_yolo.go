package main

import (
	. "./common"
	"encoding/csv"
	"flag"
	"github.com/fogleman/gg"
	"golang.org/x/image/colornames"
	"image"
	"image/draw"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// read a directory of chips and create a mosaic with bounding boxes
func main() {
	sourcedir := flag.String("source", "", "Source dir")
	targetdir := flag.String("target", "", "Output dir")

	flag.Parse()
	if *sourcedir == "" || *targetdir == "" {
		flag.Usage()
		return
	}

	os.Mkdir(*targetdir, 0755)

	files, err := ioutil.ReadDir(*sourcedir)
	if err != nil {
		log.Fatal(err)
	}

	for _, file := range files {
		if strings.HasSuffix(file.Name(), "txt") {
			imagename := strings.TrimSuffix(file.Name(), "txt") + "jpg"
			labelfile := filepath.Join(*sourcedir, file.Name())
			imagefile := filepath.Join(*sourcedir, imagename)
			outfile := filepath.Join(*targetdir, imagename)

			RenderChip(labelfile, imagefile, outfile)
		}
	}
}

func RenderChip(labelfile, imagefile, outfile string) {
	file, err := os.Open(imagefile)
	if err != nil {
		log.Fatalf("%v", err)
	}

	im, err := jpeg.Decode(file)
	if err != nil {
		log.Fatalf("%s: %v\n", imagefile, err)
	}

	f, _ := os.Open(labelfile)
	csvr := csv.NewReader(f)
	csvr.Comma = ' '
	csv, _ := csvr.ReadAll()
	labels := ReadYoloLabels(csv)

	rendered := image.NewRGBA(im.Bounds())
	draw.Draw(rendered, im.Bounds(), im, image.ZP, draw.Src)

	sz := im.Bounds().Size()
	dc := gg.NewContext(sz.X, sz.Y)

	dc.DrawImage(im, 0, 0)
	dc.SetColor(colornames.Red)

	dc.SetLineWidth(2)
	dc.SetColor(colornames.Red)
	for _, det := range labels {
		b := YoloToRect(det, sz)
		dc.DrawRectangle(float64(b.Min.X), float64(b.Min.Y), float64(b.Size().X), float64(b.Size().Y))
		dc.Stroke()
	}
	dc.SaveJPG(outfile, 75)
}

func YoloToRect(label YoloLabel, size image.Point) image.Rectangle {
	W := float64(size.X)
	H := float64(size.Y)

	dw := 1. / W
	dh := 1. / H

	w0 := label.W / dw
	h0 := label.H / dh
	xmid := label.X / dw
	ymid := label.Y / dh

	x0, x1 := xmid-w0/2., xmid+w0/2.
	y0, y1 := ymid-h0/2., ymid+h0/2.

	return image.Rectangle{
		Min: image.Point{X: int(x0), Y: int(y0)},
		Max: image.Point{X: int(x1), Y: int(y1)},
	}
}
