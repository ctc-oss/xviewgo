//
// modified version of https://github.com/sjwhitworth/golearn/blob/master/evaluation/confusion.go
//

package common

import (
	"bytes"
	"fmt"
	"sort"
	"text/tabwriter"
)

const FP CID = 0

// Truth to predictions
type Tp struct {
	T int
	P map[CID]int
}

type ConfusionMatrix map[CID]Tp

// GetConfusionMatrix builds a ConfusionMatrix from a set of reference (`ref')
// and generate (`gen') Instances.
func GetConfusionMatrix(t map[CID]int, p map[TID]Match, fp map[CID]int) (ConfusionMatrix, error) {
	ret := make(ConfusionMatrix)

	for cid, cnt := range t {
		ret[cid] = Tp{
			T: cnt,
			P: make(map[CID]int),
		}
	}

	// FPs; actual is nothing, to predicted c
	fpc := 0
	fpm := make(map[CID]int)
	for cid, cnt := range fp {
		fpc += cnt
		fpm[cid] = cnt
	}
	ret[FP] = Tp{T: fpc, P: fpm}

	// TPs; truth to predicted
	for _, m := range p {
		ret[m.T.Class].P[m.D.Class]++
	}

	return ret, nil
}

// GetTruePositives returns the number of times an entry is
// predicted successfully in a given ConfusionMatrix.
func GetTruePositives(class CID, c ConfusionMatrix) float64 {
	return float64(c[class].P[class])
}

// GetFalsePositives returns the number of times an entry is
// incorrectly predicted as having a given class.
func GetFalsePositives(class CID, c ConfusionMatrix) float64 {
	ret := 0
	for k := range c {
		if k != class {
			ret += c[k].P[class]
		}
	}
	ret += c[0].P[class]
	return float64(ret)
}

// GetFalseNegatives returns the number of times an entry is
// incorrectly predicted as something other than the given class.
func GetFalseNegatives(class CID, c ConfusionMatrix) float64 {
	ret := 0
	for k := range c[class].P {
		if k != class {
			ret += c[class].P[k]
		}
	}
	return float64(ret)
}

// GetTrueNegatives returns the number of times an entry is
// correctly predicted as something other than the given class.
func GetTrueNegatives(class CID, c ConfusionMatrix) float64 {
	ret := 0
	for k := range c {
		if k != class {
			for l := range c[k].P {
				if l != class {
					ret += c[k].P[l]
				}
			}
		}
	}
	return float64(ret)
}

// GetPrecision returns the fraction of of the total predictions
// for a given class which were correct.
func GetPrecision(class CID, c ConfusionMatrix) float64 {
	// Fraction of retrieved instances that are relevant
	truePositives := GetTruePositives(class, c)
	falsePositives := GetFalsePositives(class, c)
	return truePositives / (truePositives + falsePositives)
}

// GetRecall returns the fraction of the total occurrences of a
// given class which were predicted.
func GetRecall(class CID, c ConfusionMatrix) float64 {
	// Fraction of relevant instances that are retrieved
	truePositives := GetTruePositives(class, c)
	return truePositives / float64(c[class].T)
}

// GetF1Score computes the harmonic mean of precision and recall
// (equivalently called F-measure)
func GetF1Score(class CID, c ConfusionMatrix) float64 {
	precision := GetPrecision(class, c)
	recall := GetRecall(class, c)
	return 2 * (precision * recall) / (precision + recall)
}

// GetAccuracy computes the overall classification accuracy
// That is (number of correctly classified instances) / total instances
func GetAccuracy(c ConfusionMatrix) float64 {
	correct := 0
	total := 0
	for i := range c {
		for j := range c[i].P {
			if i == j {
				correct += c[i].P[j]
			}
			total += c[i].T
		}
	}
	return float64(correct) / float64(total)
}

// GetMicroPrecision assesses Classifier performance across
// all classes using the total true positives and false positives.
func GetMicroPrecision(c ConfusionMatrix) float64 {
	truePositives := 0.0
	falsePositives := 0.0
	for k := range c {
		truePositives += GetTruePositives(k, c)
		falsePositives += GetFalsePositives(k, c)
	}
	return truePositives / (truePositives + falsePositives)
}

// GetMacroPrecision assesses Classifier performance across all
// classes by averaging the precision measures achieved for each class.
func GetMacroPrecision(c ConfusionMatrix) float64 {
	precisionVals := 0.0
	for k := range c {
		precisionVals += GetPrecision(k, c)
	}
	return precisionVals / float64(len(c))
}

// GetMicroRecall assesses Classifier performance across all
// classes using the total true positives and false negatives.
func GetMicroRecall(c ConfusionMatrix) float64 {
	truePositives := 0.0
	falseNegatives := 0.0
	for k := range c {
		truePositives += GetTruePositives(k, c)
		falseNegatives += GetFalseNegatives(k, c)
	}
	return truePositives / (truePositives + falseNegatives)
}

// GetMacroRecall assesses Classifier performance across all classes
// by averaging the recall measures achieved for each class
func GetMacroRecall(c ConfusionMatrix) float64 {
	recallVals := 0.0
	for k := range c {
		recallVals += GetRecall(k, c)
	}
	return recallVals / float64(len(c))
}

// GetSummary returns a table of precision, recall, true positive,
// false positive, and true negatives for each class for a given
// ConfusionMatrix
func GetSummary(c ConfusionMatrix) string {
	var buffer bytes.Buffer
	w := new(tabwriter.Writer)
	w.Init(&buffer, 0, 8, 0, '\t', 0)

	fmt.Fprintln(w, "Reference Class\tTruth\tTrue Positives\tFalse Positives\tTrue Negatives\tFalse Negatives\tPrecision\tRecall\tF1 Score")
	fmt.Fprintln(w, "---------------\t-----\t--------------\t---------------\t--------------\t---------\t---------\t------\t--------")

	keys := make([]int, 0, len(c))
	for k := range c {
		if k != 0 {
			keys = append(keys, int(k))
		}
	}
	sort.Ints(keys)

	for _, ik := range keys {
		k := CID(ik)
		//==============================
		t := c[k].T
		tp := GetTruePositives(k, c)
		fp := GetFalsePositives(k, c)
		tn := GetTrueNegatives(k, c)
		fn := GetFalseNegatives(k, c)
		prec := GetPrecision(k, c)
		rec := GetRecall(k, c)
		f1 := GetF1Score(k, c)
		fmt.Fprintf(w, "%v\t%v\t%.0f\t%.0f\t%.0f\t%.0f\t%.3f\t%.3f\t%.4f\n", k, t, tp, fp, tn, fn, prec, rec, f1)
	}
	w.Flush()
	buffer.WriteString(fmt.Sprintf("False Positives:  %v\n", c[0].T))
	buffer.WriteString(fmt.Sprintf("Overall accuracy: %.4f\n", GetAccuracy(c)))

	return buffer.String()
}
