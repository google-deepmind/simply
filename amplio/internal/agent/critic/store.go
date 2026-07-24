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

package critic

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"amplio/internal/db"
)

// reportKind is the observation kind for run reports. reportObsID makes each
// iteration a distinct row (run_report-1, run_report-2, …) so history is kept,
// and re-generating a version overwrites it idempotently (INSERT OR REPLACE).
const reportKind = "run_report"

func reportObsID(version int) string { return fmt.Sprintf("%s-%d", reportKind, version) }

// AllReports returns every report for a run, ascending by version. Malformed
// rows are skipped rather than failing the whole read.
func AllReports(ctx context.Context, store db.Store, runID string) ([]*RunReport, error) {
	recs, err := store.GetObservations(ctx, runID, db.ObsFilter{Kind: reportKind})
	if err != nil {
		return nil, fmt.Errorf("get run reports: %w", err)
	}
	reports := make([]*RunReport, 0, len(recs))
	for _, rec := range recs {
		r, err := dataToReport(rec.Data)
		if err != nil {
			continue
		}
		reports = append(reports, r)
	}
	sort.Slice(reports, func(i, j int) bool { return reports[i].Version < reports[j].Version })
	return reports, nil
}

// LatestReport returns the highest-version report, or (nil, nil) if none exist.
func LatestReport(ctx context.Context, store db.Store, runID string) (*RunReport, error) {
	all, err := AllReports(ctx, store, runID)
	if err != nil || len(all) == 0 {
		return nil, err
	}
	return all[len(all)-1], nil
}

// writeReport persists a report as its versioned observation (run-level: no
// session_id / step) and denormalizes its grade onto the run row so the UI can
// show a critic grade without loading the full report.
func writeReport(ctx context.Context, store db.Store, runID string, r *RunReport) error {
	if r.CreatedAt.IsZero() {
		r.CreatedAt = time.Now().UTC()
	}
	if err := store.AppendObservation(ctx, db.ObservationRecord{
		ObsID:     reportObsID(r.Version),
		RunID:     runID,
		Kind:      reportKind,
		Data:      reportToData(r),
		CreatedAt: r.CreatedAt,
	}); err != nil {
		return err
	}
	// Cache the critic's grade on the run row. The human grade (Run.grade)
	// overrides it when set; this is the fallback the card shows otherwise.
	return store.SetRunReportGrade(ctx, runID, r.Grade)
}

// reportToData / dataToReport round-trip a report through JSON into the generic
// observation Data map (and back), so the JSON tags on RunReport define the
// stored shape without a hand-written field mapping.
func reportToData(r *RunReport) map[string]any {
	b, _ := json.Marshal(r) //nolint:errcheck // marshaling a plain struct can't fail
	var m map[string]any
	_ = json.Unmarshal(b, &m) //nolint:errcheck
	return m
}

func dataToReport(m map[string]any) (*RunReport, error) {
	b, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	var r RunReport
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, err
	}
	return &r, nil
}
