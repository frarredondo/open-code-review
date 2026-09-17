// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/alibaba/open-code-review/internal/llm"
)

func TestLLMTestCmd_HasAgentFlag(t *testing.T) {
	if llmTestCmd.Flags().Lookup("agent") == nil {
		t.Fatal("ocr llm test is missing --agent")
	}
}

func TestLLMTest_AgentResolvesConfigured(t *testing.T) {
	writeLLMTestHostAgentConfig(t, map[string]HostAgentConfig{
		"claude": {Command: "claude", Args: []string{"--foo"}, Env: []string{"FOO=bar"}},
	}, "claude-opus-4-6")

	orig := llmTestAgent
	t.Cleanup(func() { llmTestAgent = orig })
	llmTestAgent = "claude"

	ep, err := resolveLLMTestEndpoint()
	if err != nil {
		t.Fatalf("resolveLLMTestEndpoint: %v", err)
	}
	if ep.Protocol != llm.ProtocolHostAgent {
		t.Errorf("Protocol = %q, want %q", ep.Protocol, llm.ProtocolHostAgent)
	}
	if !ep.AmbientAuth {
		t.Error("AmbientAuth = false, want true")
	}
	if ep.URL != "" || ep.Token != "" {
		t.Errorf("URL/Token = %q/%q, want empty", ep.URL, ep.Token)
	}
	if ep.AgentCommand != "claude" {
		t.Errorf("AgentCommand = %q, want claude", ep.AgentCommand)
	}
	if ep.Model != "claude-opus-4-6" {
		t.Errorf("Model = %q, want claude-opus-4-6", ep.Model)
	}
}

func TestLLMTest_AgentNotConfigured(t *testing.T) {
	writeLLMTestHostAgentConfig(t, map[string]HostAgentConfig{
		"claude": {Command: "claude"},
	}, "claude-opus-4-6")

	orig := llmTestAgent
	t.Cleanup(func() { llmTestAgent = orig })
	llmTestAgent = "nope"

	_, err := resolveLLMTestEndpoint()
	if err == nil {
		t.Fatal("expected error for unconfigured agent")
	}
	if !strings.Contains(err.Error(), `agent "nope"`) || !strings.Contains(err.Error(), "host_agents") {
		t.Errorf("error = %q, want the same host_agents message as review --agent", err)
	}
}

func writeLLMTestHostAgentConfig(t *testing.T, agents map[string]HostAgentConfig, model string) {
	t.Helper()
	home := t.TempDir()
	setTestHome(t, home)
	t.Setenv("OCR_CONFIG_PATH", "")
	t.Setenv("OCR_LLM_URL", "")
	t.Setenv("OCR_LLM_TOKEN", "")
	t.Setenv("OCR_LLM_MODEL", "")
	t.Setenv("ANTHROPIC_BASE_URL", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_MODEL", "")

	cfgDir := filepath.Join(home, ".opencodereview")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := saveConfig(filepath.Join(cfgDir, "config.json"), &Config{
		Model:      model,
		HostAgents: agents,
	}); err != nil {
		t.Fatalf("saveConfig: %v", err)
	}
}
