package main

import (
	. "./common"
	"encoding/csv"
	"encoding/json"
	"flag"
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
		Id     int    `json:"feature_id"`
		Bounds string `json:"bounds_imcoords"`
		Class  int    `json:"type_id"`
	}
}

type Stats struct {
	DetectionClasses   map[CID]int
	GroundTruthClasses map[CID]int
	AveragePrecision   map[CID]float32
}

func main() {
	pFile := flag.String("predictions", "", "Path to predictions csv")
	tFile := flag.String("groundtruth", "", "Path to ground-truth geojson")
	minIou := flag.Float64("iou", .5, "IOU threshold")
	minConf := flag.Float64("confidence", .5, "Confidence threshold")

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

	stats := Stats{
		DetectionClasses:   make(map[CID]int),
		GroundTruthClasses: make(map[CID]int),
	}

	detects := make([]Detect, len(ref.Features))
	for i, splits := range predictions {
		mx, _ := strconv.Atoi(splits[0])
		my, _ := strconv.Atoi(splits[1])
		Mx, _ := strconv.Atoi(splits[2])
		My, _ := strconv.Atoi(splits[3])
		class, _ := strconv.Atoi(splits[4])
		score, _ := strconv.ParseFloat(splits[5], 32)
		detects = append(detects, Detect{
			Id: DID(i),
			Bounds: image.Rectangle{
				Min: image.Point{X: mx, Y: my},
				Max: image.Point{X: Mx, Y: My},
			},
			Class:      CID(class),
			Chip:       nil,
			Confidence: float32(score),
		})

		stats.DetectionClasses[CID(class)]++
	}

	truth := make([]Truth, len(ref.Features))
	for i, rf := range ref.Features {
		splits := strings.Split(rf.Properties.Bounds, ",")
		mx, _ := strconv.Atoi(splits[0])
		my, _ := strconv.Atoi(splits[1])
		Mx, _ := strconv.Atoi(splits[2])
		My, _ := strconv.Atoi(splits[3])
		truth[i] = Truth{
			Id: TID(rf.Properties.Id),
			Bounds: image.Rectangle{
				Min: image.Point{X: mx, Y: my},
				Max: image.Point{X: Mx, Y: My},
			},
			Class: CID(rf.Properties.Class),
		}

		stats.GroundTruthClasses[CID(rf.Properties.Class)]++
	}

	matched := make(map[TID]Match, len(truth))
	unmatched := make(map[CID]int)

	for _, d := range detects {
		if d.Confidence >= float32(*minConf) {
			found := false
			for _, t := range truth {
				if _, here := matched[t.Id]; !here {
					i := t.Bounds.Intersect(d.Bounds)
					if !i.Empty() {
						// calculate IOU;
						iz := i.Size()
						ia := iz.X * iz.Y

						dz := d.Bounds.Size()
						tz := t.Bounds.Size()
						da := dz.X * dz.Y
						ta := tz.X * tz.Y

						iou := float32(ia) / float32(da+ta-ia)
						if iou >= float32(*minIou) {
							matched[t.Id] = Match{T: t, D: d, IoU: iou}

							found = true
							break
						}
					}
				}
			}

			if !found {
				// false-positive due to non intersecting box
				// todo;; could also be a duplicate
				unmatched[d.Class]++
			}
		}
	}

	cm, _ := GetConfusionMatrix(stats.GroundTruthClasses, matched, unmatched)

	println(len(ref.Features))
	println(len(predictions))
	println(GetSummary(cm))
}
