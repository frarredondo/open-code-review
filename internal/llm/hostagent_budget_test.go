// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm_test

import (
	"context"
	"testing"

	"github.com/alibaba/open-code-review/internal/config/template"
	"github.com/alibaba/open-code-review/internal/llm"
	"github.com/alibaba/open-code-review/internal/llmloop"
	"github.com/alibaba/open-code-review/internal/session"
	"github.com/alibaba/open-code-review/internal/tool"
)

// nilUsageTransport is a HostAgentTransport that returns structured JSON
// without a usage object, matching a Claude Code CLI response that omitted it.
type nilUsageTransport struct {
	raw []byte
}

func (n *nilUsageTransport) Complete(ctx context.Context, req llm.HostAgentRequest) ([]byte, *llm.UsageInfo, error) {
	return n.raw, nil, nil
}

func TestHostAgentClient_SynthesizedUsageReachesRunnerBudgetCounters(t *testing.T) {
	client := llm.NewHostAgentClient(&nilUsageTransport{
		raw: []byte(`{"tool":"task_done","arguments":{}}`),
	})
	reg := tool.NewRegistry()
	runner := llmloop.NewRunner(llmloop.Deps{
		LLMClient:        client,
		Model:            "test-model",
		Template:         template.Template{MaxTokens: 100000, MaxToolRequestTimes: 10},
		Tools:            reg,
		CommentCollector: tool.NewCommentCollector(),
		Session:          session.New(t.TempDir(), "main", "test-model", session.SessionOptions{ReviewMode: "diff"}),
		MainToolDefs: []llm.ToolDef{{
			Type:     "function",
			Function: llm.FunctionDef{Name: "task_done", Parameters: map[string]any{"type": "object"}},
		}},
	})

	msgs := []llm.Message{llm.NewTextMessage("user", "please review this substantial patch for correctness and safety")}
	completed, _, err := runner.RunMainTask(context.Background(), msgs, "main.go")
	if err != nil {
		t.Fatalf("RunMainTask: %v", err)
	}
	if !completed {
		t.Fatal("expected task_done to complete RunMainTask")
	}
	if got := runner.TotalTokensUsed(); got == 0 {
		t.Fatal("TotalTokensUsed() = 0 after a completed turn; nil Usage silently disabled the token budget")
	}
}
