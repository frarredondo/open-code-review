// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	runAsHostAgentCLIEnv     = "_OCR_HOSTAGENT_CLI_MODE"
	hostAgentCLIArgvFileEnv  = "_OCR_HOSTAGENT_CLI_ARGV_FILE"
	hostAgentCLIArgvLogEnv   = "_OCR_HOSTAGENT_CLI_ARGV_LOG"
	hostAgentCLIStdinFileEnv = "_OCR_HOSTAGENT_CLI_STDIN_FILE"
	hostAgentCLIPidFileEnv   = "_OCR_HOSTAGENT_CLI_PID_FILE"
)

func TestMain(m *testing.M) {
	mode := os.Getenv(runAsHostAgentCLIEnv)
	if mode == "" {
		os.Exit(m.Run())
		return
	}
	runFakeHostAgentCLI(mode)
}

func runFakeHostAgentCLI(mode string) {
	args := os.Args[1:]
	if p := os.Getenv(hostAgentCLIArgvFileEnv); p != "" {
		b, _ := json.Marshal(args)
		_ = os.WriteFile(p, b, 0o600)
	}
	if p := os.Getenv(hostAgentCLIArgvLogEnv); p != "" {
		b, _ := json.Marshal(args)
		f, err := os.OpenFile(p, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
		if err == nil {
			_, _ = f.Write(append(b, '\n'))
			_ = f.Close()
		}
	}
	stdin, _ := io.ReadAll(os.Stdin)
	if p := os.Getenv(hostAgentCLIStdinFileEnv); p != "" {
		_ = os.WriteFile(p, stdin, 0o600)
	}
	if p := os.Getenv(hostAgentCLIPidFileEnv); p != "" {
		_ = os.WriteFile(p, []byte(strconv.Itoa(os.Getpid())), 0o600)
	}

	switch mode {
	case "happy":
		os.Stdout.WriteString(`{"type":"result","subtype":"success","is_error":false,"session_id":"s1","total_cost_usd":0.01,"structured_output":{"text":"ok","n":1},"usage":{"input_tokens":10,"output_tokens":4,"cache_creation_input_tokens":2,"cache_read_input_tokens":3},"modelUsage":{"ignored/model":{"inputTokens":999,"outputTokens":999}}}` + "\n")
		os.Exit(0)
	case "error-success-subtype":
		os.Stdout.WriteString(`{"is_error":true,"subtype":"success","terminal_reason":"api_error","result":"Not logged in · Please run /login"}` + "\n")
		os.Exit(1)
	case "no-structured-output":
		os.Stdout.WriteString(`{"subtype":"success","is_error":false,"result":"hello"}` + "\n")
		os.Exit(0)
	case "null-structured-output":
		os.Stdout.WriteString(`{"subtype":"success","is_error":false,"structured_output":null}` + "\n")
		os.Exit(0)
	case "exit-empty-stderr":
		os.Exit(1)
	case "not-json":
		os.Stdout.WriteString("this is not json\n")
		os.Exit(0)
	case "hang":
		time.Sleep(30 * time.Second)
		os.Exit(0)
	case "huge":
		os.Stdout.Write(bytes.Repeat([]byte("x"), 10000))
		os.Exit(0)
	default:
		os.Stderr.WriteString("unknown fake host-agent CLI mode\n")
		os.Exit(2)
	}
}

func TestCLITransport_HappyPathUsage(t *testing.T) {
	tr, _, stdinFile := newTestCLI(t, "happy")
	req := sampleHostAgentRequest()
	req.Prompt = "review the diff"
	raw, usage, err := tr.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("structured_output is not JSON %q: %v", raw, err)
	}
	if got["text"] != "ok" || got["n"] != float64(1) {
		t.Errorf("structured_output = %#v, want text=ok n=1", got)
	}
	if usage == nil {
		t.Fatal("usage is nil")
	}
	if usage.PromptTokens != 10 || usage.CompletionTokens != 4 {
		t.Errorf("tokens = prompt %d completion %d, want 10 and 4", usage.PromptTokens, usage.CompletionTokens)
	}
	if usage.CacheWriteTokens != 2 || usage.CacheReadTokens != 3 {
		t.Errorf("cache = write %d read %d, want 2 and 3", usage.CacheWriteTokens, usage.CacheReadTokens)
	}
	if usage.TotalTokens != 14 {
		t.Errorf("TotalTokens = %d, want 14 (input+output)", usage.TotalTokens)
	}
	stdin, err := os.ReadFile(stdinFile)
	if err != nil {
		t.Fatalf("stdin dump: %v", err)
	}
	if string(stdin) != "review the diff" {
		t.Errorf("stdin = %q, want the prompt", stdin)
	}
}

func TestCLITransport_IsErrorDespiteSuccessSubtype(t *testing.T) {
	tr, _, _ := newTestCLI(t, "error-success-subtype")
	_, _, err := tr.Complete(context.Background(), sampleHostAgentRequest())
	if err == nil {
		t.Fatal("is_error true with subtype success returned nil error")
	}
	if !strings.Contains(err.Error(), "Not logged in") && !strings.Contains(err.Error(), "api_error") {
		t.Errorf("error %q does not mention the CLI result or terminal_reason", err)
	}
}

func TestCLITransport_SuccessWithoutStructuredOutput(t *testing.T) {
	tr, _, _ := newTestCLI(t, "no-structured-output")
	_, _, err := tr.Complete(context.Background(), sampleHostAgentRequest())
	if err == nil {
		t.Fatal("success with no structured_output returned nil error")
	}
	if !strings.Contains(err.Error(), "structured_output") {
		t.Errorf("error %q does not mention structured_output", err)
	}
}

func TestCLITransport_NullStructuredOutput(t *testing.T) {
	tr, _, _ := newTestCLI(t, "null-structured-output")
	_, _, err := tr.Complete(context.Background(), sampleHostAgentRequest())
	if err == nil {
		t.Fatal("success with null structured_output returned nil error")
	}
	if !strings.Contains(err.Error(), "structured_output") {
		t.Errorf("error %q does not mention structured_output", err)
	}
}

func TestCLITransport_NonZeroExitEmptyStderr(t *testing.T) {
	tr, _, _ := newTestCLI(t, "exit-empty-stderr")
	_, _, err := tr.Complete(context.Background(), sampleHostAgentRequest())
	if err == nil {
		t.Fatal("non-zero exit with empty stderr returned nil error")
	}
	if !strings.Contains(err.Error(), "1") {
		t.Errorf("error %q does not mention the exit status", err)
	}
}

func TestCLITransport_StdoutNotJSON(t *testing.T) {
	tr, _, _ := newTestCLI(t, "not-json")
	_, _, err := tr.Complete(context.Background(), sampleHostAgentRequest())
	if err == nil {
		t.Fatal("non-JSON stdout returned nil error")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "json") {
		t.Errorf("error %q does not mention JSON", err)
	}
}

func TestCLITransport_ContextCanceledKillsChild(t *testing.T) {
	origDelay := hostAgentCLIWaitDelay
	hostAgentCLIWaitDelay = 100 * time.Millisecond
	t.Cleanup(func() { hostAgentCLIWaitDelay = origDelay })

	tr, _, _ := newTestCLI(t, "hang")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	start := time.Now()
	go func() {
		_, _, err := tr.Complete(ctx, sampleHostAgentRequest())
		done <- err
	}()
	time.Sleep(50 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Complete did not return after cancel; child likely outlived the context")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("took %s after cancel; child outlived the context", elapsed)
	}
}

func TestCLITransport_OutputExceedsCap(t *testing.T) {
	orig := hostAgentCLIMaxOutput
	hostAgentCLIMaxOutput = 1024
	t.Cleanup(func() { hostAgentCLIMaxOutput = orig })

	tr, _, _ := newTestCLI(t, "huge")
	_, _, err := tr.Complete(context.Background(), sampleHostAgentRequest())
	if err == nil {
		t.Fatal("output over cap returned nil error")
	}
	if !strings.Contains(err.Error(), "cap") {
		t.Errorf("error %q does not mention the output cap", err)
	}
}

func TestCLITransport_ArgvResumeSystemPromptAndTools(t *testing.T) {
	t.Run("flags always present and resume/system omitted when empty", func(t *testing.T) {
		tr, argvFile, _ := newTestCLI(t, "happy")
		req := sampleHostAgentRequest()
		req.System = ""
		req.SessionID = ""
		req.Model = "opus"
		if _, _, err := tr.Complete(context.Background(), req); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		args := readArgv(t, argvFile)
		assertFlagValue(t, args, "--tools", "")
		assertFlagValue(t, args, "--output-format", "json")
		assertFlagValue(t, args, "--model", "opus")
		assertHasFlag(t, args, "--bare")
		assertHasFlag(t, args, "-p")
		assertHasFlag(t, args, "--json-schema")
		if hasFlag(args, "--resume") {
			t.Errorf("argv %v has --resume with empty SessionID", args)
		}
		if hasFlag(args, "--session-id") {
			t.Errorf("argv %v has --session-id with empty SessionID", args)
		}
		if hasFlag(args, "--system-prompt") {
			t.Errorf("argv %v has --system-prompt with empty System", args)
		}
		if hasFlag(args, "--max-turns") {
			t.Errorf("argv %v has --max-turns; do not depend on a version-skewed flag", args)
		}
		if containsString(args, req.Prompt) {
			t.Errorf("prompt was passed as argv; it must go on stdin")
		}
	})
	t.Run("session-id and system-prompt present on first call", func(t *testing.T) {
		tr, argvFile, _ := newTestCLI(t, "happy")
		req := sampleHostAgentRequest()
		req.System = "You are a reviewer."
		req.SessionID = "sess-1"
		if _, _, err := tr.Complete(context.Background(), req); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		args := readArgv(t, argvFile)
		assertSessionStart(t, args, "sess-1")
		assertFlagValue(t, args, "--system-prompt", "You are a reviewer.")
		assertFlagValue(t, args, "--tools", "")
	})
	t.Run("model omitted when empty", func(t *testing.T) {
		tr, argvFile, _ := newTestCLI(t, "happy")
		req := sampleHostAgentRequest()
		req.Model = ""
		if _, _, err := tr.Complete(context.Background(), req); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		args := readArgv(t, argvFile)
		if hasFlag(args, "--model") {
			t.Errorf("argv %v has --model with empty Model", args)
		}
	})
}

func TestCLITransport_SessionIDFirstThenResume(t *testing.T) {
	const (
		idA = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
		idB = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
	)

	t.Run("first call uses session-id, second uses resume", func(t *testing.T) {
		tr, argvFile, _ := newTestCLI(t, "happy")
		req := sampleHostAgentRequest()
		req.SessionID = idA

		if _, _, err := tr.Complete(context.Background(), req); err != nil {
			t.Fatalf("first Complete: %v", err)
		}
		assertSessionStart(t, readArgv(t, argvFile), idA)

		if _, _, err := tr.Complete(context.Background(), req); err != nil {
			t.Fatalf("second Complete: %v", err)
		}
		assertSessionResume(t, readArgv(t, argvFile), idA)
	})

	t.Run("distinct ids each start with session-id", func(t *testing.T) {
		tr, argvFile, _ := newTestCLI(t, "happy")
		reqA := sampleHostAgentRequest()
		reqA.SessionID = idA
		reqB := sampleHostAgentRequest()
		reqB.SessionID = idB

		if _, _, err := tr.Complete(context.Background(), reqA); err != nil {
			t.Fatalf("Complete A: %v", err)
		}
		assertSessionStart(t, readArgv(t, argvFile), idA)

		if _, _, err := tr.Complete(context.Background(), reqB); err != nil {
			t.Fatalf("Complete B: %v", err)
		}
		assertSessionStart(t, readArgv(t, argvFile), idB)

		if _, _, err := tr.Complete(context.Background(), reqA); err != nil {
			t.Fatalf("second Complete A: %v", err)
		}
		assertSessionResume(t, readArgv(t, argvFile), idA)
	})

	t.Run("concurrent distinct ids both start", func(t *testing.T) {
		tr, argvFile, _ := newTestCLI(t, "happy")
		ids := []string{idA, idB}
		errs := make([]error, len(ids))
		var wg sync.WaitGroup
		wg.Add(len(ids))
		for i, id := range ids {
			i, id := i, id
			go func() {
				defer wg.Done()
				req := sampleHostAgentRequest()
				req.SessionID = id
				_, _, errs[i] = tr.Complete(context.Background(), req)
			}()
		}
		wg.Wait()
		for i, err := range errs {
			if err != nil {
				t.Fatalf("Complete[%d]: %v", i, err)
			}
		}
		dumps := readArgvLog(t, argvFile)
		if len(dumps) != 2 {
			t.Fatalf("argv log has %d entries, want 2", len(dumps))
		}
		seen := map[string]string{}
		for _, args := range dumps {
			flag, value := sessionFlag(args)
			if flag == "" {
				t.Errorf("argv %v has no session flag", args)
				continue
			}
			seen[value] = flag
		}
		if seen[idA] != "--session-id" {
			t.Errorf("id A flag = %q, want --session-id (log=%v)", seen[idA], dumps)
		}
		if seen[idB] != "--session-id" {
			t.Errorf("id B flag = %q, want --session-id (log=%v)", seen[idB], dumps)
		}
	})

	t.Run("concurrent same id one start one resume", func(t *testing.T) {
		tr, argvFile, _ := newTestCLI(t, "happy")
		errs := make([]error, 2)
		var wg sync.WaitGroup
		wg.Add(2)
		for i := range errs {
			i := i
			go func() {
				defer wg.Done()
				req := sampleHostAgentRequest()
				req.SessionID = idA
				_, _, errs[i] = tr.Complete(context.Background(), req)
			}()
		}
		wg.Wait()
		for i, err := range errs {
			if err != nil {
				t.Fatalf("Complete[%d]: %v", i, err)
			}
		}
		dumps := readArgvLog(t, argvFile)
		if len(dumps) != 2 {
			t.Fatalf("argv log has %d entries, want 2", len(dumps))
		}
		starts, resumes := 0, 0
		for _, args := range dumps {
			switch flag, value := sessionFlag(args); flag {
			case "--session-id":
				if value != idA {
					t.Errorf("--session-id = %q, want %s", value, idA)
				}
				starts++
			case "--resume":
				if value != idA {
					t.Errorf("--resume = %q, want %s", value, idA)
				}
				resumes++
			default:
				t.Errorf("argv %v has session flag %q, want --session-id or --resume", args, flag)
			}
		}
		if starts != 1 || resumes != 1 {
			t.Errorf("starts=%d resumes=%d, want 1 and 1 (log=%v)", starts, resumes, dumps)
		}
	})
}

func sampleHostAgentRequest() HostAgentRequest {
	return HostAgentRequest{
		Prompt: "review the diff",
		Schema: map[string]any{"type": "object"},
		Model:  "opus",
	}
}

func newTestCLI(t *testing.T, mode string) (*cliTransport, string, string) {
	t.Helper()
	dir := t.TempDir()
	argvFile := filepath.Join(dir, "argv.json")
	stdinFile := filepath.Join(dir, "stdin.txt")
	tr := newCLITransport(os.Args[0], nil)
	tr.extraEnv = []string{
		runAsHostAgentCLIEnv + "=" + mode,
		hostAgentCLIArgvFileEnv + "=" + argvFile,
		hostAgentCLIArgvLogEnv + "=" + filepath.Join(dir, "argv.log"),
		hostAgentCLIStdinFileEnv + "=" + stdinFile,
		hostAgentCLIPidFileEnv + "=" + filepath.Join(dir, "pid"),
	}
	return tr, argvFile, stdinFile
}

func readArgv(t *testing.T, path string) []string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read argv dump: %v", err)
	}
	var args []string
	if err := json.Unmarshal(b, &args); err != nil {
		t.Fatalf("argv dump: %v", err)
	}
	return args
}

func readArgvLog(t *testing.T, argvFile string) [][]string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(filepath.Dir(argvFile), "argv.log"))
	if err != nil {
		t.Fatalf("read argv log: %v", err)
	}
	var out [][]string
	for _, line := range bytes.Split(b, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var args []string
		if err := json.Unmarshal(line, &args); err != nil {
			t.Fatalf("argv log: %v", err)
		}
		out = append(out, args)
	}
	return out
}

func sessionFlag(args []string) (flag, value string) {
	hasSID := hasFlag(args, "--session-id")
	hasResume := hasFlag(args, "--resume")
	switch {
	case hasSID && hasResume:
		return "both", ""
	case hasSID:
		return "--session-id", flagValue(args, "--session-id")
	case hasResume:
		return "--resume", flagValue(args, "--resume")
	default:
		return "", ""
	}
}

func flagValue(args []string, flag string) string {
	for i, a := range args {
		if a == flag {
			if i+1 < len(args) {
				return args[i+1]
			}
			return ""
		}
	}
	return ""
}

func assertSessionStart(t *testing.T, args []string, id string) {
	t.Helper()
	assertFlagValue(t, args, "--session-id", id)
	if hasFlag(args, "--resume") {
		t.Errorf("argv %v has --resume; first call for an id must use --session-id", args)
	}
}

func assertSessionResume(t *testing.T, args []string, id string) {
	t.Helper()
	assertFlagValue(t, args, "--resume", id)
	if hasFlag(args, "--session-id") {
		t.Errorf("argv %v has --session-id; subsequent call must use --resume", args)
	}
}

func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

func assertHasFlag(t *testing.T, args []string, flag string) {
	t.Helper()
	if !hasFlag(args, flag) {
		t.Errorf("argv %v missing %s", args, flag)
	}
}

func assertFlagValue(t *testing.T, args []string, flag, want string) {
	t.Helper()
	for i, a := range args {
		if a == flag {
			if i+1 >= len(args) {
				t.Errorf("flag %s has no value in %v", flag, args)
				return
			}
			if args[i+1] != want {
				t.Errorf("flag %s = %q, want %q", flag, args[i+1], want)
			}
			return
		}
	}
	t.Errorf("argv %v missing %s", args, flag)
}

func containsString(args []string, s string) bool {
	for _, a := range args {
		if a == s {
			return true
		}
	}
	return false
}
