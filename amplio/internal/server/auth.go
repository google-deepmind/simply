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
	"crypto/subtle"
	"net/http"
	"strings"
)

// authCookie is the default web-auth cookie name (holds the token after the
// magic-link exchange; HttpOnly so JS can't read it). Per-instance servers
// override it via SetCookieName — see cookieKey.
const authCookie = "amplio_auth"

// cookieKey is the auth cookie name this instance uses: a per-instance override
// if set (so co-located servers on different ports don't clobber each other —
// cookies ignore the port), else the default.
func (s *Server) cookieKey() string {
	if s.cookieName != "" {
		return s.cookieName
	}
	return authCookie
}

// bearerOrQueryToken reads the token a caller explicitly presents: a ?token=
// query param (EventSource / <img> can't set headers) or a Bearer header (CLI).
// It deliberately ignores the cookie — used by login, which must validate the
// freshly-presented token, not a stale cookie.
func bearerOrQueryToken(r *http.Request) string {
	if t := r.URL.Query().Get("token"); t != "" {
		return t
	}
	return strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
}

// authed reports whether the request carries the valid token via cookie (web),
// query, or Bearer (CLI). An empty configured token disables auth (tests /
// explicitly-open local use): everything is authed.
func (s *Server) authed(r *http.Request) bool {
	if s.token == "" {
		return true
	}
	got := bearerOrQueryToken(r)
	if got == "" {
		if c, err := r.Cookie(s.cookieKey()); err == nil {
			got = c.Value
		}
	}
	return subtle.ConstantTimeCompare([]byte(got), []byte(s.token)) == 1
}

// requireAuth gates a mutating handler. Reads are wired unguarded (the readonly
// share view); only writes go through here.
func (s *Server) requireAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !s.authed(r) {
			writeErr(w, http.StatusUnauthorized, "invalid or missing token")
			return
		}
		next(w, r)
	}
}

// handleAuthLogin exchanges a valid presented token for an HttpOnly cookie, so
// the frontend can strip ?token= from the URL and later token-less shared links
// resolve to the readonly view. SameSite=Lax gives basic CSRF protection for
// cross-site writes. The Secure flag is set when the server has TLS configured
// (see SetSecureCookie); plain-HTTP deployments leave it unset so browsers
// still send the cookie over the unencrypted connection.
func (s *Server) handleAuthLogin(w http.ResponseWriter, r *http.Request) {
	if s.token != "" {
		if subtle.ConstantTimeCompare([]byte(bearerOrQueryToken(r)), []byte(s.token)) != 1 {
			writeErr(w, http.StatusUnauthorized, "invalid token")
			return
		}
		http.SetCookie(w, &http.Cookie{
			Name:     s.cookieKey(),
			Value:    s.token,
			Path:     "/",
			HttpOnly: true,
			Secure:   s.secureCookie,
			SameSite: http.SameSiteLaxMode,
			MaxAge:   30 * 24 * 3600, // 30 days; refreshed on each login
		})
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// handleAuth reports whether the caller may write (drives the readonly UI) and
// the server owner's username (for the readonly banner).
func (s *Server) handleAuth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"authed": s.authed(r),
		"user":   s.owner,
	})
}
