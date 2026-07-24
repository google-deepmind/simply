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

package server

import (
	"net/http"
	"strconv"
	"strings"
	"time"
)

// recallResults mirrors what an agent's recall_search returns: two ranked
// sections keyed by typed handle.
type recallResults struct {
	Skills  []recallSkillHit  `json:"skills"`
	Lessons []recallLessonHit `json:"lessons"`
}

type recallSkillHit struct {
	Handle      string `json:"handle"`
	Name        string `json:"name"`
	Description string `json:"description"`
}

type recallLessonHit struct {
	Handle      string `json:"handle"`
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
	Score       int    `json:"score"`
	LoadedCount int    `json:"loaded_count"`
}

// handleRecallSearch ranks the skill + lesson corpora for a query, exactly as
// agents do. Empty query → empty sections (the page can still list lessons).
func (s *Server) handleRecallSearch(w http.ResponseWriter, r *http.Request) {
	q := strings.TrimSpace(r.URL.Query().Get("q"))
	k := 15
	if v := r.URL.Query().Get("k"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			k = n
		}
	}
	out := recallResults{Skills: []recallSkillHit{}, Lessons: []recallLessonHit{}}
	if q == "" {
		writeJSON(w, http.StatusOK, out)
		return
	}
	if s.skillIndex != nil && s.skillIndex.IsBuilt() {
		hits, err := s.skillIndex.Search(r.Context(), q, k)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		for _, h := range hits {
			out.Skills = append(out.Skills, recallSkillHit{
				Handle: "skill:" + h.Entry.Name, Name: h.Entry.Name, Description: h.Entry.Description,
			})
		}
	}
	if s.lessonIndex != nil && s.lessonIndex.IsBuilt() {
		hits, err := s.lessonIndex.Search(r.Context(), q, k)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		for _, h := range hits {
			out.Lessons = append(out.Lessons, recallLessonHit{
				Handle: "lesson:" + h.Lesson.LessonID, ID: h.Lesson.LessonID, Title: h.Lesson.Title,
				Description: h.Lesson.Description, Score: h.Lesson.Score, LoadedCount: h.Lesson.LoadedCount,
			})
		}
	}
	writeJSON(w, http.StatusOK, out)
}

// recallItem is the full body of one corpus entry (the recall_load content).
type recallItem struct {
	Kind        string `json:"kind"` // "skill" | "lesson"
	Name        string `json:"name,omitempty"`
	Path        string `json:"path,omitempty"`
	ID          string `json:"id,omitempty"`
	Title       string `json:"title,omitempty"`
	Description string `json:"description,omitempty"`
	Body        string `json:"body"`
	Score       int    `json:"score,omitempty"`
	LoadedCount int    `json:"loaded_count,omitempty"`
	SourceRun   string `json:"source_run,omitempty"`
}

func (s *Server) handleRecallItem(w http.ResponseWriter, r *http.Request) {
	handle := strings.TrimSpace(r.URL.Query().Get("handle"))
	if name, ok := strings.CutPrefix(handle, "skill:"); ok {
		if s.skillIndex == nil || !s.skillIndex.IsBuilt() {
			writeErr(w, http.StatusNotFound, "skill recall unavailable")
			return
		}
		e, ok := s.skillIndex.Load(name)
		if !ok {
			writeErr(w, http.StatusNotFound, "skill not found")
			return
		}
		writeJSON(w, http.StatusOK, recallItem{Kind: "skill", Name: e.Name, Path: e.Path, Description: e.Description, Body: e.Body})
		return
	}
	if id, ok := strings.CutPrefix(handle, "lesson:"); ok {
		if s.lessonIndex == nil || !s.lessonIndex.IsBuilt() {
			writeErr(w, http.StatusNotFound, "lesson recall unavailable")
			return
		}
		l, ok := s.lessonIndex.Load(id)
		if !ok {
			writeErr(w, http.StatusNotFound, "lesson not found")
			return
		}
		writeJSON(w, http.StatusOK, recallItem{
			Kind: "lesson", ID: l.LessonID, Title: l.Title, Description: l.Description, Body: l.Body,
			Score: l.Score, LoadedCount: l.LoadedCount, SourceRun: l.SourceRunID,
		})
		return
	}
	writeErr(w, http.StatusBadRequest, "invalid handle; expected skill:<name> or lesson:<id>")
}

type lessonSummary struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Score       int       `json:"score"`
	LoadedCount int       `json:"loaded_count"`
	SourceRun   string    `json:"source_run"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// handleListLessons lists the whole mined lesson corpus (newest first) — reads
// the DB directly, so it works even if the index isn't built.
func (s *Server) handleListLessons(w http.ResponseWriter, r *http.Request) {
	recs, err := s.store.ListAllLessons(r.Context())
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	out := make([]lessonSummary, 0, len(recs))
	for i := len(recs) - 1; i >= 0; i-- { // ListAllLessons is ascending; show newest first
		l := recs[i]
		out = append(out, lessonSummary{
			ID: l.LessonID, Title: l.Title, Description: l.Description, Score: l.Score,
			LoadedCount: l.LoadedCount, SourceRun: l.SourceRunID, CreatedAt: l.CreatedAt, UpdatedAt: l.UpdatedAt,
		})
	}
	writeJSON(w, http.StatusOK, out)
}
