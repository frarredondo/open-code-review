// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"strings"
	"testing"
)

func TestLookupProvider_HostAgentAmbientAuth(t *testing.T) {
	p, ok := LookupProvider("host-agent")
	if !ok {
		t.Fatal("LookupProvider(host-agent) returned false, want true")
	}
	if !p.AmbientAuth {
		t.Error("host-agent preset AmbientAuth = false, want true")
	}
	if p.Protocol != ProtocolHostAgent {
		t.Errorf("Protocol = %q, want %q", p.Protocol, ProtocolHostAgent)
	}
	if p.BaseURL != "" {
		t.Errorf("BaseURL = %q, want empty", p.BaseURL)
	}
	if p.EnvVar != "" {
		t.Errorf("EnvVar = %q, want empty", p.EnvVar)
	}
}

func TestResolveEndpointWithOptions_AgentConfigured(t *testing.T) {
	clearAllEnv(t)
	path, _ := writeResolverConfig(t, configFile{
		Model: "claude-opus-4-6",
		HostAgents: map[string]hostAgentFileConfig{
			"claude": {
				Command: "claude",
				Args:    []string{"--foo"},
				Env:     []string{"FOO=bar"},
			},
		},
	})

	ep, err := ResolveEndpointWithOptions(path, ResolveOptions{Agent: "claude"})
	if err != nil {
		t.Fatalf("ResolveEndpointWithOptions: %v", err)
	}
	if ep.Protocol != ProtocolHostAgent {
		t.Errorf("Protocol = %q, want %q", ep.Protocol, ProtocolHostAgent)
	}
	if !ep.AmbientAuth {
		t.Error("AmbientAuth = false, want true")
	}
	if ep.URL != "" {
		t.Errorf("URL = %q, want empty", ep.URL)
	}
	if ep.Token != "" {
		t.Errorf("Token = %q, want empty", ep.Token)
	}
	if ep.Model != "claude-opus-4-6" {
		t.Errorf("Model = %q, want claude-opus-4-6", ep.Model)
	}
	if ep.AgentCommand != "claude" {
		t.Errorf("AgentCommand = %q, want claude", ep.AgentCommand)
	}
	if len(ep.AgentArgs) != 1 || ep.AgentArgs[0] != "--foo" {
		t.Errorf("AgentArgs = %v, want [--foo]", ep.AgentArgs)
	}
	if len(ep.AgentEnv) != 1 || ep.AgentEnv[0] != "FOO=bar" {
		t.Errorf("AgentEnv = %v, want [FOO=bar]", ep.AgentEnv)
	}
}

func TestResolveEndpointWithOptions_AgentModelOverride(t *testing.T) {
	clearAllEnv(t)
	path, _ := writeResolverConfig(t, configFile{
		Model: "claude-sonnet-4-6",
		HostAgents: map[string]hostAgentFileConfig{
			"claude": {Command: "claude"},
		},
	})

	ep, err := ResolveEndpointWithOptions(path, ResolveOptions{Agent: "claude", Model: "claude-opus-4-6"})
	if err != nil {
		t.Fatalf("ResolveEndpointWithOptions: %v", err)
	}
	if ep.Model != "claude-opus-4-6" {
		t.Errorf("Model = %q, want claude-opus-4-6", ep.Model)
	}
}

func TestResolveEndpointWithOptions_AgentNotConfigured(t *testing.T) {
	clearAllEnv(t)
	path, _ := writeResolverConfig(t, configFile{
		Model: "claude-opus-4-6",
		HostAgents: map[string]hostAgentFileConfig{
			"claude": {Command: "claude"},
		},
	})

	_, err := ResolveEndpointWithOptions(path, ResolveOptions{Agent: "missing"})
	if err == nil {
		t.Fatal("expected error for unconfigured agent")
	}
	if !strings.Contains(err.Error(), `agent "missing"`) || !strings.Contains(err.Error(), "host_agents") {
		t.Errorf("error = %q, want it to name the agent and host_agents", err)
	}
}

func TestResolveEndpointWithOptions_AgentAndProviderMutuallyExclusive(t *testing.T) {
	clearAllEnv(t)
	path, _ := writeResolverConfig(t, configFile{
		Provider: "anthropic",
		Model:    "claude-sonnet-4-6",
		Providers: map[string]providerEntryConfig{
			"anthropic": {APIKey: "sk-ant-test", Model: "claude-sonnet-4-6"},
		},
		HostAgents: map[string]hostAgentFileConfig{
			"claude": {Command: "claude"},
		},
	})

	_, err := ResolveEndpointWithOptions(path, ResolveOptions{Agent: "claude", Provider: "anthropic"})
	if err == nil {
		t.Fatal("expected error when --agent and --provider are both set")
	}
	if !strings.Contains(err.Error(), "--agent") || !strings.Contains(err.Error(), "--provider") {
		t.Errorf("error = %q, want it to mention --agent and --provider", err)
	}
}

// TestResolveEndpointWithOptions_ProviderUnaffectedByHostAgents is a regression
// guard: a normal provider resolution must still produce the same endpoint it
// did before host-agent wiring, even when host_agents is also present in the
// file.
func TestResolveEndpointWithOptions_ProviderUnaffectedByHostAgents(t *testing.T) {
	clearAllEnv(t)
	path, _ := writeResolverConfig(t, configFile{
		Provider: "anthropic",
		Providers: map[string]providerEntryConfig{
			"anthropic": {APIKey: "sk-ant-test", Model: "claude-sonnet-4-6"},
		},
		HostAgents: map[string]hostAgentFileConfig{
			"claude": {Command: "claude", Args: []string{"--should-not-be-used"}},
		},
	})

	ep, err := ResolveEndpoint(path)
	if err != nil {
		t.Fatalf("ResolveEndpoint: %v", err)
	}
	if ep.Protocol != ProtocolAnthropic {
		t.Errorf("Protocol = %q, want %q", ep.Protocol, ProtocolAnthropic)
	}
	if ep.Token != "sk-ant-test" {
		t.Errorf("Token = %q, want sk-ant-test", ep.Token)
	}
	if ep.Model != "claude-sonnet-4-6" {
		t.Errorf("Model = %q, want claude-sonnet-4-6", ep.Model)
	}
	if ep.Source != "provider:anthropic" {
		t.Errorf("Source = %q, want provider:anthropic", ep.Source)
	}
	if ep.AgentCommand != "" {
		t.Errorf("AgentCommand = %q, want empty on a provider endpoint", ep.AgentCommand)
	}
	if ep.AmbientAuth {
		t.Error("AmbientAuth = true, want false for anthropic")
	}
}
