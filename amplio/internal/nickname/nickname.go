// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package nickname

import (
	"fmt"
	"math/rand/v2"
	"strings"
)

// Reserved session IDs for per-run singletons.
const (
	RootAgent = "main-agent"
	Chatbot   = "chatty-bot"
)

var adjectives = [...]string{
	"ace", "agile", "amber", "ample", "balmy", "bold", "brave", "breezy",
	"bright", "brisk", "busy", "calm", "chatty", "cheery", "chill", "clever",
	"cool", "cosmic", "cozy", "crisp", "dapper", "deft", "dewy", "dreamy",
	"dusky", "eager", "easy", "fair", "fancy", "fast", "feisty", "firm",
	"free", "fresh", "frosty", "gentle", "glad", "golden", "grand", "happy",
	"hardy", "jaunty", "jolly", "jovial", "kind", "level", "lithe",
	"lively", "lofty", "loyal", "lucky", "lush", "magic", "main", "mellow",
	"merry", "mighty", "mild", "misty", "neat", "nimble", "open", "peachy",
	"peppy", "perky", "plain", "plucky", "polite", "posh", "proud", "pure",
	"quick", "quiet", "rapid", "ready", "regal", "rosy", "royal", "rustic",
	"safe", "sharp", "silver", "sleek", "smart", "smooth", "snappy", "snug",
	"soft", "solid", "spry", "steady", "still", "sturdy", "suave", "sunny",
	"swift", "tame", "tidy", "true", "vivid",
}

var nouns = [...]string{
	"agent", "ant", "badger", "bear", "beaver", "bee", "bison", "boar",
	"bot", "bunny", "camel", "cat", "cobra", "condor", "coyote", "crab",
	"crane", "crow", "deer", "dingo", "dolphin", "dove", "duck",
	"eagle", "egret", "elk", "emu", "falcon", "ferret", "finch", "fish",
	"fox", "frog", "gecko", "gnu", "goat", "goose", "grouse", "gull",
	"hare", "hawk", "hen", "heron", "hippo", "horse", "husky", "ibex",
	"ibis", "iguana", "jackal", "jay", "kite", "koala", "lark", "lion",
	"lizard", "llama", "lynx", "marmot", "moose", "moth", "newt", "oryx",
	"otter", "owl", "panda", "parrot", "pelican", "penguin", "pigeon",
	"pony", "possum", "puffin", "quail", "rabbit", "raven", "robin",
	"salmon", "seal", "sheep", "shrew", "sloth", "snail", "snake",
	"sparrow", "spider", "squid", "stag", "stoat", "stork", "swallow",
	"swan", "tern", "thrush", "tiger", "toad", "trout", "turtle", "viper",
	"vole",
}

// reservedAdj and reservedNoun are derived from the reserved nicknames.
var (
	reservedAdj  map[string]bool
	reservedNoun map[string]bool
)

func init() {
	reserved := []string{RootAgent, Chatbot}
	reservedAdj = make(map[string]bool, len(reserved))
	reservedNoun = make(map[string]bool, len(reserved))
	for _, name := range reserved {
		parts := strings.SplitN(name, "-", 2)
		reservedAdj[parts[0]] = true
		reservedNoun[parts[1]] = true
	}
}

const (
	fallbackSuffixLen      = 3
	fallbackMaxAttempts    = 8
	fallbackSuffixAlphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
)

// PickUnique returns an adj-noun nickname not in used.
// Pass nil for rng to use a default source. The caller is responsible for
// race-checking via the storage layer's strict Insert.
func PickUnique(used map[string]bool, rng *rand.Rand) string {
	if rng == nil {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())) //nolint:gosec // nicknames are not security-critical
	}

	// Build dynamic pool excluding reserved components.
	var available []string
	for _, a := range adjectives {
		if reservedAdj[a] {
			continue
		}
		for _, n := range nouns {
			if reservedNoun[n] {
				continue
			}
			name := a + "-" + n
			if !used[name] {
				available = append(available, name)
			}
		}
	}

	if len(available) > 0 {
		return available[rng.IntN(len(available))]
	}

	// Pool exhausted — fallback to suffix mode.
	for range fallbackMaxAttempts {
		a := adjectives[rng.IntN(len(adjectives))]
		n := nouns[rng.IntN(len(nouns))]
		suffix := make([]byte, fallbackSuffixLen)
		for i := range suffix {
			suffix[i] = fallbackSuffixAlphabet[rng.IntN(len(fallbackSuffixAlphabet))]
		}
		candidate := fmt.Sprintf("%s-%s-%s", a, n, suffix)
		if !used[candidate] {
			return candidate
		}
	}

	panic(fmt.Sprintf(
		"failed to allocate a unique fallback-suffix nickname after %d attempts",
		fallbackMaxAttempts,
	))
}

// IsReserved reports whether name is one of the reserved session IDs.
func IsReserved(name string) bool {
	return name == RootAgent || name == Chatbot
}
