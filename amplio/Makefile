# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY: build dev test test-race test-integration lint fmt vuln cover frontend-build notices clean

# Build tags applied to every go invocation. The internal tag switches in
# environment-specific implementations (workspace backend, credential probe,
# extra UI). The OSS Makefile drops the tag automatically — see
# copybara/copy.bara.sky.
#
# Raw `go build ./...` (no Makefile, no TAGS) compiles the OSS subset and is
# the safety fallback for anyone bypassing this Makefile.
TAGS ?=

# Integration tests carry their own tag plus `internal`. Go takes only the
# last -tags flag, so we can't compose; the Copybara mirror drops the
# `internal` half.
INTEGRATION_TAGS ?= -tags=integration

# Build the amplio binary with embedded frontend.
build: frontend-build
	go build $(TAGS) -o amplio ./cmd/amplio

# Dev workflow instructions (run in separate terminals).
dev:
	@echo "Run in separate terminals:"
	@echo "  go run $(TAGS) ./cmd/amplio serve  # backend on :26759 (prints a URL+token)"
	@echo "  cd frontend && npm run dev         # Vite on :5173, proxies /api -> :26759"
	@echo "Open the Vite URL with the token from the serve banner: http://localhost:5173/?token=<token>"

# Run all tests.
test:
	go test $(TAGS) ./...

# Run tests with race detector (default for CI).
test-race:
	go test -race $(TAGS) ./...

# Run integration tests (real LLM calls, skipped in CI).
test-integration:
	go test -race $(INTEGRATION_TAGS) ./...

# Run linters.
lint:
	go tool golangci-lint run ./...

# Format all Go source files.
fmt:
	go fmt ./...

# Check dependencies for known vulnerabilities.
vuln:
	go tool govulncheck $(TAGS) ./...

# Generate test coverage report.
cover:
	go test $(TAGS) -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

# Build the frontend for production embedding into internal/server/webdist/client
# (git-ignored). Installs JS deps when missing OR stale: npm writes
# node_modules/.package-lock.json on install, so if the committed lockfile is
# newer than that marker (e.g. a pull added deps), reinstall. Guarding only on
# `[ -d node_modules ]` would silently skip new deps and fail the build with an
# unresolved import.
frontend-build:
	cd frontend && { [ ! package-lock.json -nt node_modules/.package-lock.json ] || npm ci; } && npm run build

# Regenerate THIRD_PARTY_NOTICES.txt (attribution for deps redistributed in
# release artifacts). Requires node_modules present (npm ci) for the npm side.
notices:
	cd frontend && { [ ! package-lock.json -nt node_modules/.package-lock.json ] || npm ci; }
	./scripts/gen-third-party-notices.sh

# Remove build artifacts.
clean:
	rm -f amplio coverage.out coverage.html
	rm -rf frontend/build internal/server/webdist/client
