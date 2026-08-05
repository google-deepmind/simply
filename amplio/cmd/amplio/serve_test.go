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

package main

import (
	"net"
	"os"
	"testing"
)

func TestListenAddr(t *testing.T) {
	const cfg = "0.0.0.0:26759"
	tests := []struct {
		name           string
		flag, env, cfg string
		want           string
	}{
		{"default from config", "", "", cfg, cfg},
		{"env overrides config", "", "0.0.0.0:1111", cfg, "0.0.0.0:1111"},
		{"flag overrides env and config", "0.0.0.0:2222", "0.0.0.0:1111", cfg, "0.0.0.0:2222"},
		{"flag overrides config (no env)", "0.0.0.0:2222", "", cfg, "0.0.0.0:2222"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := listenAddr(tc.flag, tc.env, tc.cfg); got != tc.want {
				t.Errorf("listenAddr(%q,%q,%q) = %q, want %q", tc.flag, tc.env, tc.cfg, got, tc.want)
			}
		})
	}
}

func TestBannerHosts(t *testing.T) {
	hn, _ := os.Hostname()
	// On a misconfigured machine where os.Hostname() returns "localhost",
	// the wildcard case should collapse to a single entry rather than
	// duplicate "localhost:port".
	expectCollapsed := hn == "" || hn == "localhost"

	tests := []struct {
		name string
		ip   net.IP
		port int
		// want is the expected slice for non-wildcard cases. For wildcard
		// cases, leave nil and the test checks structure separately
		// (hostname is machine-dependent).
		want     []string
		wildcard bool
	}{
		{name: "ipv4 wildcard", ip: net.IPv4zero, port: 26759, wildcard: true},
		{name: "ipv6 wildcard", ip: net.IPv6unspecified, port: 26759, wildcard: true},
		{name: "ipv4 loopback", ip: net.IPv4(127, 0, 0, 1), port: 26759,
			want: []string{"localhost:26759"}},
		{name: "ipv4 loopback non-default", ip: net.IPv4(127, 0, 0, 42), port: 8080,
			want: []string{"localhost:8080"}},
		{name: "ipv6 loopback", ip: net.IPv6loopback, port: 26759,
			want: []string{"localhost:26759"}},
		{name: "specific ipv4", ip: net.IPv4(192, 168, 1, 5), port: 26759,
			want: []string{"192.168.1.5:26759"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := bannerHosts(&net.TCPAddr{IP: tc.ip, Port: tc.port})
			if !tc.wildcard {
				if !slicesEqual(got, tc.want) {
					t.Errorf("bannerHosts(%v:%d) = %q, want %q", tc.ip, tc.port, got, tc.want)
				}
				return
			}
			// Wildcard: hostname-dependent. Either 2 entries [<hn>:port,
			// localhost:port] or (when hostname is missing/"localhost") 1
			// entry [localhost:port].
			wantSuffix := ":" + itoa(tc.port)
			wantLocal := "localhost" + wantSuffix
			if expectCollapsed {
				if len(got) != 1 || got[0] != wantLocal {
					t.Errorf("bannerHosts(wildcard %v) = %q, want [%q] (collapsed)", tc.ip, got, wantLocal)
				}
				return
			}
			if len(got) != 2 {
				t.Fatalf("bannerHosts(wildcard %v) = %q, want 2 entries", tc.ip, got)
			}
			if got[0] != hn+wantSuffix {
				t.Errorf("bannerHosts(wildcard %v)[0] = %q, want %q", tc.ip, got[0], hn+wantSuffix)
			}
			if got[1] != wantLocal {
				t.Errorf("bannerHosts(wildcard %v)[1] = %q, want %q", tc.ip, got[1], wantLocal)
			}
		})
	}
}

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// itoa avoids importing strconv just for one call site.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// TestLendAddr_Precedence: the lending listener follows the same flag > env >
// config rule as --listen. It was config-only when introduced, which made it
// the one address setting you could not set from a script.
func TestLendAddr_Precedence(t *testing.T) {
	for _, tc := range []struct{ flag, env, cfg, want string }{
		{"", "", "", ""},                                             // absent everywhere = lending off
		{"", "", "127.0.0.1:1", "127.0.0.1:1"},                       // config
		{"", "127.0.0.1:2", "127.0.0.1:1", "127.0.0.1:2"},            // env beats config
		{"127.0.0.1:3", "127.0.0.1:2", "127.0.0.1:1", "127.0.0.1:3"}, // flag beats both
	} {
		if got := addrFrom(tc.flag, tc.env, tc.cfg); got != tc.want {
			t.Errorf("addrFrom(%q,%q,%q) = %q, want %q", tc.flag, tc.env, tc.cfg, got, tc.want)
		}
	}
}
