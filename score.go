package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	. "github.com/jw3/example-tensorflow-golang/common"
	"image"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
)

type FeatureCollection struct {
	Features []Feature
}

type Feature struct {
	Properties struct {
		Bounds string `json:"bounds_imcoords"`
		Class  int    `json:"type_id"`
	}
}

func main() {
	pFile := flag.String("predictions", "", "Path to predictions csv")
	tFile := flag.String("groundtruth", "", "Path to ground-truth geojson")

	flag.Parse()
	if *pFile == "" || *tFile == "" {
		flag.Usage()
		return
	}

	_, err := ioutil.ReadFile(*pFile)
	if err != nil {
		log.Fatal(err)
	}

	f, _ := os.Open(*pFile)
	csvr := csv.NewReader(f)
	csvr.Comma = ' '
	predictions, _ := csvr.ReadAll()

	tbytes, err := ioutil.ReadFile(*tFile)
	if err != nil {
		log.Fatal(err)
	}

	var ref FeatureCollection
	json.Unmarshal(tbytes, &ref)

	// (xmin,ymin,xmax,ymax)

	println(len(ref.Features))

	detects := make([]Detect, len(ref.Features))
	for _, splits := range predictions {
		mx, _ := strconv.Atoi(splits[0])
		my, _ := strconv.Atoi(splits[1])
		Mx, _ := strconv.Atoi(splits[2])
		My, _ := strconv.Atoi(splits[3])
		class, _ := strconv.Atoi(splits[4])
		score, _ := strconv.ParseFloat(splits[5], 32)
		detects = append(detects, Detect{
			Bounds: image.Rectangle{
				Min: image.Point{X: mx, Y: my},
				Max: image.Point{X: Mx, Y: My},
			},
			Class:      class,
			Chip:       nil,
			Confidence: float32(score),
		})
	}

	println(len(predictions))

	truth := make([]Detect, len(predictions))
	for _, rf := range ref.Features {
		splits := strings.Split(rf.Properties.Bounds, ",")
		mx, _ := strconv.Atoi(splits[0])
		my, _ := strconv.Atoi(splits[1])
		Mx, _ := strconv.Atoi(splits[2])
		My, _ := strconv.Atoi(splits[3])
		truth = append(truth, Detect{
			Bounds: image.Rectangle{
				Min: image.Point{X: mx, Y: my},
				Max: image.Point{X: Mx, Y: My},
			},
			Class:      rf.Properties.Class,
			Chip:       nil,
			Confidence: 0,
		})
	}

	// calculate IOU;
	intersects := 0
	for _, d := range detects {
		if d.Confidence > .3 {
			for _, t := range truth {
				i := t.Bounds.Intersect(d.Bounds)
				if !i.Empty() {
					iz := i.Size()
					ia := iz.X * iz.Y

					dz := d.Bounds.Size()
					tz := t.Bounds.Size()
					da := dz.X * dz.Y
					ta := tz.X * tz.Y

					iou := float32(ia) / float32(da+ta-ia)

					if iou > .1 {
						fmt.Printf("%.2f\n", iou)
					}

					intersects++
				}
			}
		}
	}

	print(intersects)
}
