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
	"log"
	"os"
)

func main() {
	imagefile := flag.String("image", "", "Path of a JPEG-image to extract labels for")
	pFile := flag.String("predictions", "", "Path to predictions csv, or - for stdin")
	minConf := flag.Float64("confidence", .5, "Confidence threshold")
	debugmode := flag.Bool("debug", false, "Enable debug mode")

	const (
		H, W = 544, 544
	)

	flag.Parse()
	if *pFile == "" {
		flag.Usage()
		return
	}

	file, err := os.Open(*imagefile)
	if err != nil {
		log.Fatalf("%v", err)
	}

	im, err := jpeg.Decode(file)
	if err != nil {
		log.Fatalf("%s: %v\n", *imagefile, err)
	}

	f, _ := os.Open(*pFile)
	csvr := csv.NewReader(f)
	csvr.Comma = ' '
	predictions, _ := csvr.ReadAll()
	detects := ReadDetects(predictions)

	rendered := image.NewRGBA(im.Bounds())
	draw.Draw(rendered, im.Bounds(), im, image.ZP, draw.Src)

	sz := im.Bounds().Size()
	dc := gg.NewContext(sz.X, sz.Y)

	dc.DrawImage(im, 0, 0)
	dc.SetColor(colornames.Red)

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
	dc.SavePNG("/tmp/rendered.png")
}
