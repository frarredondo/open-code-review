// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
)

// Transport sends a prompt plus JSON Schema to a host-agent inference backend
// and returns the raw structured bytes. Client depends only on this interface.
type Transport interface {
	Complete(ctx context.Context, prompt string, schema map[string]any) (raw []byte, usage *UsageInfo, err error)
}

// Client is an LLMClient that turns ChatRequest tools into a JSON Schema,
// delegates inference to a Transport, and maps the structured result back.
type Client struct {
	transport Transport
}

// NewClient returns a Client that uses transport for inference.
func NewClient(transport Transport) *Client {
	return &Client{transport: transport}
}

func (c *Client) CompletionsWithCtx(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	raw, usage, err := c.transport.Complete(ctx, promptFromMessages(req.Messages), schemaForTools(req.Tools))
	if err != nil {
		return nil, err
	}
	resp, err := responseToChat(raw, req.Model)
	if err != nil {
		return nil, err
	}
	resp.Usage = usage
	return resp, nil
}

func promptFromMessages(messages []Message) string {
	var b strings.Builder
	for i, m := range messages {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(m.Role)
		b.WriteString(": ")
		b.WriteString(m.ExtractText())
		for _, tc := range m.ToolCalls {
			b.WriteString("\ntool_call ")
			b.WriteString(tc.Function.Name)
			b.WriteString(" ")
			b.WriteString(tc.Function.Arguments)
		}
		if m.ToolCallID != "" {
			b.WriteString(" tool_call_id=")
			b.WriteString(m.ToolCallID)
		}
	}
	return b.String()
}

func schemaForTools(tools []ToolDef) map[string]any {
	branches := make([]any, 0, len(tools)+1)
	for _, tool := range tools {
		var arguments any = tool.Function.Parameters
		if tool.Function.Parameters == nil {
			arguments = nil
		}
		branches = append(branches, map[string]any{
			"type": "object",
			"properties": map[string]any{
				"tool":      map[string]any{"const": tool.Function.Name},
				"arguments": arguments,
			},
			"required": []string{"tool", "arguments"},
		})
	}
	branches = append(branches, map[string]any{
		"type": "object",
		"properties": map[string]any{
			"text": map[string]any{"type": "string"},
		},
		"required": []string{"text"},
	})
	return map[string]any{
		"$schema": "http://json-schema.org/draft-07/schema#",
		"oneOf":   branches,
	}
}

func responseToChat(raw []byte, model string) (*ChatResponse, error) {
	var payload any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, fmt.Errorf("host-agent response: %w", err)
	}
	choice, err := choiceFromPayload(payload)
	if err != nil {
		return nil, err
	}
	return &ChatResponse{
		Model:   model,
		Choices: []Choice{choice},
	}, nil
}

func choiceFromPayload(payload any) (Choice, error) {
	switch v := payload.(type) {
	case []any:
		if len(v) == 0 {
			return Choice{}, fmt.Errorf("host-agent response: empty tool list")
		}
		calls := make([]ToolCall, 0, len(v))
		for _, item := range v {
			call, err := toolCallFrom(item)
			if err != nil {
				return Choice{}, err
			}
			calls = append(calls, call)
		}
		return Choice{
			FinishReason: "tool_calls",
			Message: ResponseMessage{
				Role:      "assistant",
				ToolCalls: calls,
			},
		}, nil
	case map[string]any:
		if _, ok := v["tool"]; ok {
			call, err := toolCallFrom(v)
			if err != nil {
				return Choice{}, err
			}
			return Choice{
				FinishReason: "tool_calls",
				Message: ResponseMessage{
					Role:      "assistant",
					ToolCalls: []ToolCall{call},
				},
			}, nil
		}
		text, ok := v["text"]
		if !ok {
			return Choice{}, fmt.Errorf("host-agent response: unrecognized object")
		}
		s, ok := text.(string)
		if !ok {
			return Choice{}, fmt.Errorf("host-agent response: text must be a string")
		}
		return Choice{
			FinishReason: "stop",
			Message: ResponseMessage{
				Role:    "assistant",
				Content: &s,
			},
		}, nil
	default:
		return Choice{}, fmt.Errorf("host-agent response: unrecognized JSON")
	}
}

func toolCallFrom(item any) (ToolCall, error) {
	m, ok := item.(map[string]any)
	if !ok {
		return ToolCall{}, fmt.Errorf("host-agent response: tool call must be an object")
	}
	name, _ := m["tool"].(string)
	if name == "" {
		return ToolCall{}, fmt.Errorf("host-agent response: missing tool name")
	}
	args := m["arguments"]
	if args == nil {
		args = map[string]any{}
	}
	encoded, err := json.Marshal(args)
	if err != nil {
		return ToolCall{}, fmt.Errorf("host-agent response: marshal arguments: %w", err)
	}
	return ToolCall{
		ID:   uuid.New().String(),
		Type: "function",
		Function: FunctionCall{
			Name:      name,
			Arguments: string(encoded),
		},
	}, nil
}
