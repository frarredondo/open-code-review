// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// hostAgentCLIMaxOutput caps how much of the CLI stdout we buffer. A review
// response is large but bounded; refusing further writes makes the child die
// of SIGPIPE instead of growing the heap. Package var so tests can shrink it.
var hostAgentCLIMaxOutput = 16 << 20

// hostAgentCLIWaitDelay bounds how long Wait keeps waiting on the child's
// stdout pipe after ctx cancellation. Package var so tests can shrink it.
var hostAgentCLIWaitDelay = 5 * time.Second

const hostAgentCLIStderrMax = 64 << 10

// cliTransport runs a host-agent CLI (Claude Code) as a one-shot subprocess.
// One instance is shared across concurrent group subtasks, so started
// session ids are guarded by mu.
type cliTransport struct {
	command   string
	extraArgs []string
	extraEnv  []string

	mu      sync.Mutex
	started map[string]struct{}
}

func newCLITransport(command string, extraArgs []string) *cliTransport {
	return &cliTransport{command: command, extraArgs: extraArgs}
}

func (t *cliTransport) Complete(ctx context.Context, req HostAgentRequest) ([]byte, *UsageInfo, error) {
	schemaPath, err := writeHostAgentSchemaFile(req.Schema)
	if err != nil {
		return nil, nil, err
	}
	defer os.Remove(schemaPath)

	args := t.buildArgs(req, schemaPath)
	// Unlike MCP's NewClient, this subprocess is one-shot: ctx bounds the
	// whole run so cancel kills the CLI. WaitDelay then unblocks Wait if a
	// pipe-holding grandchild outlives the kill.
	cmd := exec.CommandContext(ctx, t.command, args...)
	cmd.Env = append(os.Environ(), t.extraEnv...)
	cmd.Stdin = strings.NewReader(req.Prompt)
	stdout := &cappedBuffer{max: hostAgentCLIMaxOutput}
	stderr := &cappedBuffer{max: hostAgentCLIStderrMax}
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.WaitDelay = hostAgentCLIWaitDelay

	runErr := cmd.Run()
	if ctx.Err() != nil {
		return nil, nil, fmt.Errorf("host-agent CLI: %w", ctx.Err())
	}
	if stdout.overflow {
		return nil, nil, fmt.Errorf("host-agent CLI output exceeds cap")
	}

	parsed, parseErr := parseHostAgentCLIStdout(stdout.buf.Bytes())
	if parseErr == nil && parsed.IsError {
		return nil, nil, fmt.Errorf("%s", parsed.errorMessage())
	}
	if runErr != nil {
		return nil, nil, formatCLIExit(runErr, stderr.buf.Bytes())
	}
	if parseErr != nil {
		return nil, nil, fmt.Errorf("host-agent CLI stdout is not JSON: %w", parseErr)
	}
	raw := parsed.StructuredOutput
	if len(bytes.TrimSpace(raw)) == 0 || string(raw) == "null" {
		return nil, nil, fmt.Errorf("host-agent CLI success response missing structured_output")
	}
	return append([]byte(nil), raw...), usageFromCLI(parsed.Usage), nil
}

func (t *cliTransport) buildArgs(req HostAgentRequest, schemaPath string) []string {
	// req.MaxTokens has no CLI equivalent: `claude --help` on v2.1.274 lists
	// no --max-tokens flag. The cap is advisory for this transport;
	// enforcement lives in the aggregate token budget instead.
	args := make([]string, 0, len(t.extraArgs)+16)
	args = append(args, t.extraArgs...)
	args = append(args,
		"--bare", "-p",
		"--output-format", "json",
		"--json-schema", schemaPath,
		"--tools", "",
	)
	if req.Model != "" {
		args = append(args, "--model", req.Model)
	}
	if req.System != "" {
		args = append(args, "--system-prompt", req.System)
	}
	if req.SessionID != "" {
		// First call for an id uses --session-id so the harness creates a
		// conversation under OCR's UUID (`claude --help` on v2.1.274:
		// "Use a specific session ID for the conversation (must be a
		// valid UUID)"). Later calls use --resume, which then names a
		// session that exists.
		args = append(args, t.sessionFlag(req.SessionID), req.SessionID)
	}
	return args
}

func (t *cliTransport) sessionFlag(id string) string {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.started == nil {
		t.started = make(map[string]struct{})
	}
	if _, ok := t.started[id]; ok {
		return "--resume"
	}
	t.started[id] = struct{}{}
	return "--session-id"
}

func writeHostAgentSchemaFile(schema map[string]any) (string, error) {
	if schema == nil {
		schema = map[string]any{}
	}
	f, err := os.CreateTemp("", "ocr-hostagent-schema-*.json")
	if err != nil {
		return "", fmt.Errorf("host-agent CLI schema temp file: %w", err)
	}
	path := f.Name()
	enc := json.NewEncoder(f)
	if err := enc.Encode(schema); err != nil {
		f.Close()
		os.Remove(path)
		return "", fmt.Errorf("host-agent CLI schema temp file: %w", err)
	}
	if err := f.Close(); err != nil {
		os.Remove(path)
		return "", fmt.Errorf("host-agent CLI schema temp file: %w", err)
	}
	return path, nil
}

// hostAgentCLIResult is a permissive decode of --output-format json.
// Unknown keys are ignored: the full key set is undocumented and version-unstable.
type hostAgentCLIResult struct {
	Type             string             `json:"type"`
	Subtype          string             `json:"subtype"`
	IsError          bool               `json:"is_error"`
	Result           string             `json:"result"`
	StructuredOutput json.RawMessage    `json:"structured_output"`
	SessionID        string             `json:"session_id"`
	TotalCostUSD     float64            `json:"total_cost_usd"`
	Usage            *hostAgentCLIUsage `json:"usage"`
	TerminalReason   string             `json:"terminal_reason"`
}

type hostAgentCLIUsage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
}

func parseHostAgentCLIStdout(raw []byte) (hostAgentCLIResult, error) {
	var parsed hostAgentCLIResult
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return hostAgentCLIResult{}, err
	}
	return parsed, nil
}

func (r hostAgentCLIResult) errorMessage() string {
	switch {
	case r.Result != "" && r.TerminalReason != "":
		return fmt.Sprintf("host-agent CLI error (%s): %s", r.TerminalReason, r.Result)
	case r.Result != "":
		return "host-agent CLI error: " + r.Result
	case r.TerminalReason != "":
		return "host-agent CLI error: " + r.TerminalReason
	default:
		return "host-agent CLI reported is_error"
	}
}

func usageFromCLI(u *hostAgentCLIUsage) *UsageInfo {
	if u == nil {
		return nil
	}
	return &UsageInfo{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
		CacheWriteTokens: u.CacheCreationInputTokens,
		CacheReadTokens:  u.CacheReadInputTokens,
		TotalTokens:      u.InputTokens + u.OutputTokens,
	}
}

func formatCLIExit(err error, stderr []byte) error {
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		if len(bytes.TrimSpace(stderr)) == 0 {
			return fmt.Errorf("host-agent CLI exited with status %d (empty stderr)", ee.ExitCode())
		}
		return fmt.Errorf("host-agent CLI exited with status %d: %s", ee.ExitCode(), bytes.TrimSpace(stderr))
	}
	return fmt.Errorf("host-agent CLI: %w", err)
}
