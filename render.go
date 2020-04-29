package main

import (
	. "./common"
	"encoding/csv"
	"flag"
	"fmt"
	"github.com/fogleman/gg"
	"golang.org/x/image/colornames"
	"image"
	"image/draw"
	"log"
	"os"
)

func main() {
	imagefile := flag.String("image", "", "Path of a JPEG-image to extract labels for")
	pFile := flag.String("predictions", "-", "Path to predictions csv, or - for stdin")
	minConf := flag.Float64("confidence", .5, "Confidence threshold")
	debugmode := flag.Bool("debug", false, "Enable debug mode")
	outfile := flag.String("outfile", "/tmp/rendered.png", "Path to write rendered image file")

	const (
		H, W = 544, 544
	)

	flag.Parse()
	if *pFile == "" || *imagefile == "" {
		flag.Usage()
		return
	}

	im, err := LoadJpeg(*imagefile)
	if err != nil {
		log.Fatalf("%v", err)
	}

	var f *os.File
	if *pFile == "-" {
		f = os.Stdin
	} else {
		f, err = os.Open(*pFile)
		if err != nil {
			log.Fatalf("%s: %v\n", *pFile, err)
		}
	}

	csvr := csv.NewReader(f)
	csvr.Comma = ' '
	predictions, _ := csvr.ReadAll()
	detects := ReadDetects(predictions)
	log.Println("detections: ", len(predictions))

	rendered := image.NewRGBA(im.Bounds())
	draw.Draw(rendered, im.Bounds(), im, image.ZP, draw.Src)

	sz := im.Bounds().Size()
	dc := gg.NewContext(sz.X, sz.Y)
	dc.DrawImage(im, 0, 0)

	if *debugmode {
		dc.SetLineWidth(.1)
		dc.SetColor(colornames.Green)
		for x := 0; x < sz.X/W; x++ {
			for y := 0; y < sz.Y/H; y++ {
				dc.DrawRectangle(float64(x*W), float64(y*H), float64(x*W+W), float64(y*H+H))
				dc.Stroke()
			}
		}
	}

	dc.SetLineWidth(2)
	dc.SetColor(colornames.Red)
	for _, det := range detects {
		if det.Confidence > float32(*minConf) {
			b := det.Bounds
			dc.DrawRectangle(float64(b.Min.X), float64(b.Min.Y), float64(b.Size().X), float64(b.Size().Y))
			dc.Stroke()
		}
	}

	if err := dc.SavePNG(*outfile); err != nil {
		log.Fatalf("%s: %v\n", *pFile, err)
	}
	log.Println(fmt.Sprint("rendered to file://", *outfile))
}
