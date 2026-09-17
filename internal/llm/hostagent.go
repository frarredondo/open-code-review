// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import "context"

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
	return &ChatResponse{Model: req.Model}, nil
}

func schemaForTools(tools []ToolDef) map[string]any {
	return map[string]any{}
}

func responseToChat(raw []byte, model string) (*ChatResponse, error) {
	return &ChatResponse{Model: model}, nil
}
