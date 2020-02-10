package common

import (
	"strconv"
)

func ReadDetects(predictions [][]string) []Detect {
	detects := make([]Detect, 0)
	for i, splits := range predictions {
		class, _ := strconv.Atoi(splits[4])
		score, _ := strconv.ParseFloat(splits[5], 32)
		detects = append(detects, Detect{
			Id: DID(i),
			Bounds: SplitToRect(splits),
			Class:      CID(class),
			Chip:       nil,
			Confidence: float32(score),
		})
	}
	return detects
}
