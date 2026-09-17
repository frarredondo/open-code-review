// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

type fakeTransport struct {
	raw    []byte
	usage  *UsageInfo
	err    error
	prompt string
	schema map[string]any
}

func (f *fakeTransport) Complete(ctx context.Context, prompt string, schema map[string]any) ([]byte, *UsageInfo, error) {
	f.prompt = prompt
	f.schema = schema
	return f.raw, f.usage, f.err
}

func TestSchemaForTools_Draft07OneOf(t *testing.T) {
	schema := schemaForTools(nil)
	s, _ := schema["$schema"].(string)
	if !strings.Contains(s, "draft-07") {
		t.Errorf("$schema = %q, want a draft-07 schema URI", s)
	}
	if _, ok := schema["oneOf"]; !ok {
		t.Fatal("schema missing oneOf")
	}
}

func TestSchemaForTools_EveryToolDefShapeGetsABranch(t *testing.T) {
	objectProps := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string"},
		},
		"required": []any{"path"},
	}
	emptyObject := map[string]any{"type": "object"}
	nested := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"$comment":             "must survive",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "search query",
			},
		},
	}

	tools := []ToolDef{
		{Type: "function", Function: FunctionDef{Name: "file_read", Description: "read a file", Parameters: objectProps}},
		{Type: "function", Function: FunctionDef{Name: "task_done", Parameters: emptyObject}},
		{Type: "function", Function: FunctionDef{Name: "code_search", Parameters: nested}},
		{Type: "function", Function: FunctionDef{Name: "noop"}},
	}

	schema := schemaForTools(tools)
	branches := oneOfBranches(t, schema)
	if len(branches) != len(tools)+1 {
		t.Fatalf("oneOf len = %d, want %d (one per tool plus text)", len(branches), len(tools)+1)
	}

	assertToolBranch(t, schema, "file_read", objectProps)
	assertToolBranch(t, schema, "task_done", emptyObject)
	assertToolBranch(t, schema, "code_search", nested)
	assertToolBranch(t, schema, "noop", nil)
	assertTextBranch(t, schema)
}

func TestSchemaForTools_ParametersPassedThroughUnmodified(t *testing.T) {
	params := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string"},
		},
	}
	tools := []ToolDef{{Type: "function", Function: FunctionDef{Name: "file_read", Parameters: params}}}

	schema := schemaForTools(tools)
	got := toolArguments(t, schema, "file_read")
	if !sameMap(got, params) {
		t.Fatal("arguments schema was rebuilt; Parameters must be passed through unmodified")
	}

	params["x-marker"] = "yes"
	if got["x-marker"] != "yes" {
		t.Fatal("mutating Parameters after schemaForTools did not affect the branch; Parameters was copied")
	}
}

func TestResponseToChat_ToolBranchOneChoice(t *testing.T) {
	raw := []byte(`{"tool":"file_read","arguments":{"path":"main.go"}}`)
	resp, err := responseToChat(raw, "host-model")
	if err != nil {
		t.Fatalf("responseToChat: %v", err)
	}
	assertOneChoice(t, resp)
	if resp.Model != "host-model" {
		t.Errorf("Model = %q, want host-model", resp.Model)
	}
	msg := resp.Choices[0].Message
	if msg.Native != (NativeTurn{}) {
		t.Errorf("Native = %+v, want zero value", msg.Native)
	}
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(msg.ToolCalls))
	}
	call := msg.ToolCalls[0]
	if call.ID == "" {
		t.Error("ToolCall.ID is empty")
	}
	if call.Type != "function" {
		t.Errorf("Type = %q, want function", call.Type)
	}
	if call.Function.Name != "file_read" {
		t.Errorf("Name = %q, want file_read", call.Function.Name)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		t.Fatalf("Arguments %q is not JSON: %v", call.Function.Arguments, err)
	}
	if args["path"] != "main.go" {
		t.Errorf("arguments path = %v, want main.go", args["path"])
	}
}

func TestResponseToChat_TextBranchZeroToolCallsNonNilContent(t *testing.T) {
	raw := []byte(`{"text":"looks good"}`)
	resp, err := responseToChat(raw, "host-model")
	if err != nil {
		t.Fatalf("responseToChat: %v", err)
	}
	assertOneChoice(t, resp)
	msg := resp.Choices[0].Message
	if len(msg.ToolCalls) != 0 {
		t.Errorf("ToolCalls = %v, want none", msg.ToolCalls)
	}
	if msg.Content == nil {
		t.Fatal("Content is nil, want non-nil")
	}
	if *msg.Content != "looks good" {
		t.Errorf("Content = %q, want looks good", *msg.Content)
	}
	if msg.Native != (NativeTurn{}) {
		t.Errorf("Native = %+v, want zero value", msg.Native)
	}
}

func TestResponseToChat_MultiToolCallUniqueIDs(t *testing.T) {
	raw := []byte(`[
		{"tool":"file_read","arguments":{"path":"a.go"}},
		{"tool":"file_find","arguments":{"glob":"*.go"}}
	]`)
	resp, err := responseToChat(raw, "host-model")
	if err != nil {
		t.Fatalf("responseToChat: %v", err)
	}
	assertOneChoice(t, resp)
	calls := resp.Choices[0].Message.ToolCalls
	if len(calls) != 2 {
		t.Fatalf("ToolCalls len = %d, want 2", len(calls))
	}
	if calls[0].Function.Name != "file_read" || calls[1].Function.Name != "file_find" {
		t.Errorf("names = %q, %q", calls[0].Function.Name, calls[1].Function.Name)
	}
	if calls[0].ID == "" || calls[1].ID == "" {
		t.Fatalf("IDs must be non-empty, got %q and %q", calls[0].ID, calls[1].ID)
	}
	if calls[0].ID == calls[1].ID {
		t.Fatalf("duplicate ToolCall.ID %q", calls[0].ID)
	}
	if calls[0].Type != "function" || calls[1].Type != "function" {
		t.Errorf("types = %q, %q, want function", calls[0].Type, calls[1].Type)
	}
}

func TestResponseToChat_MalformedJSONErrorNotEmptyChoices(t *testing.T) {
	resp, err := responseToChat([]byte(`{not json`), "host-model")
	if err == nil {
		if resp != nil && len(resp.Choices) == 0 {
			t.Fatal("malformed JSON returned empty-Choices success")
		}
		t.Fatal("malformed JSON returned success, want error")
	}
	if resp != nil && len(resp.Choices) == 0 {
		t.Fatal("error response still carried empty Choices; callers treat that as a recorded error")
	}
}

func TestResponseToChat_UnrecognizedObjectError(t *testing.T) {
	resp, err := responseToChat([]byte(`{"foo":1}`), "host-model")
	if err == nil {
		if resp != nil && len(resp.Choices) == 0 {
			t.Fatal("unrecognized object returned empty-Choices success")
		}
		t.Fatal("unrecognized object returned success, want error")
	}
}

func TestClient_CompletionsMapsToolCall(t *testing.T) {
	ft := &fakeTransport{raw: []byte(`{"tool":"file_read","arguments":{"path":"x.go"}}`)}
	c := NewClient(ft)
	resp, err := c.CompletionsWithCtx(context.Background(), ChatRequest{
		Model: "opus",
		Messages: []Message{
			{Role: "user", Content: "read x.go"},
		},
		Tools: []ToolDef{{
			Type: "function",
			Function: FunctionDef{
				Name:       "file_read",
				Parameters: map[string]any{"type": "object"},
			},
		}},
	})
	if err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	assertOneChoice(t, resp)
	if resp.Model != "opus" {
		t.Errorf("Model = %q, want opus", resp.Model)
	}
	if len(resp.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(resp.Choices[0].Message.ToolCalls))
	}
	if resp.Choices[0].Message.Native != (NativeTurn{}) {
		t.Errorf("Native = %+v, want zero value", resp.Choices[0].Message.Native)
	}
}

func TestClient_CompletionsMapsText(t *testing.T) {
	ft := &fakeTransport{raw: []byte(`{"text":"done"}`)}
	c := NewClient(ft)
	resp, err := c.CompletionsWithCtx(context.Background(), ChatRequest{
		Model:    "opus",
		Messages: []Message{{Role: "user", Content: "summarize"}},
	})
	if err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	assertOneChoice(t, resp)
	msg := resp.Choices[0].Message
	if len(msg.ToolCalls) != 0 {
		t.Errorf("ToolCalls = %v, want none", msg.ToolCalls)
	}
	if msg.Content == nil || *msg.Content != "done" {
		t.Errorf("Content = %v, want done", msg.Content)
	}
}

func TestClient_CompletionsMalformedJSON(t *testing.T) {
	ft := &fakeTransport{raw: []byte(`{`)}
	c := NewClient(ft)
	resp, err := c.CompletionsWithCtx(context.Background(), ChatRequest{
		Model:    "opus",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		if resp != nil && len(resp.Choices) == 0 {
			t.Fatal("malformed JSON returned empty-Choices success")
		}
		t.Fatal("malformed JSON returned success, want error")
	}
}

func TestClient_TransportError(t *testing.T) {
	ft := &fakeTransport{err: errors.New("cli failed")}
	c := NewClient(ft)
	_, err := c.CompletionsWithCtx(context.Background(), ChatRequest{
		Model:    "opus",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected transport error")
	}
	if !strings.Contains(err.Error(), "cli failed") {
		t.Errorf("error = %v, want it to wrap cli failed", err)
	}
}

func TestClient_PassesUsageThrough(t *testing.T) {
	usage := &UsageInfo{PromptTokens: 11, CompletionTokens: 7, TotalTokens: 18, CacheReadTokens: 3}
	ft := &fakeTransport{raw: []byte(`{"text":"ok"}`), usage: usage}
	c := NewClient(ft)
	resp, err := c.CompletionsWithCtx(context.Background(), ChatRequest{
		Model:    "opus",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	if resp.Usage != usage {
		t.Errorf("Usage was not passed through unchanged: got %+v", resp.Usage)
	}
}

func TestClient_ForwardsSchemaAndPrompt(t *testing.T) {
	params := map[string]any{"type": "object"}
	tools := []ToolDef{{Type: "function", Function: FunctionDef{Name: "file_read", Parameters: params}}}
	ft := &fakeTransport{raw: []byte(`{"text":"ok"}`)}
	c := NewClient(ft)
	_, err := c.CompletionsWithCtx(context.Background(), ChatRequest{
		Model:    "opus",
		Messages: []Message{{Role: "user", Content: "review the diff"}},
		Tools:    tools,
	})
	if err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	if !strings.Contains(ft.prompt, "review the diff") {
		t.Errorf("prompt %q does not contain the user message", ft.prompt)
	}
	want := schemaForTools(tools)
	if !reflect.DeepEqual(ft.schema, want) {
		t.Errorf("transport schema = %#v, want %#v", ft.schema, want)
	}
}

func assertOneChoice(t *testing.T, resp *ChatResponse) {
	t.Helper()
	if resp == nil {
		t.Fatal("response is nil")
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("len(Choices) = %d, want 1", len(resp.Choices))
	}
}

func oneOfBranches(t *testing.T, schema map[string]any) []map[string]any {
	t.Helper()
	raw, ok := schema["oneOf"]
	if !ok {
		t.Fatal("schema missing oneOf")
	}
	switch items := raw.(type) {
	case []any:
		out := make([]map[string]any, len(items))
		for i, item := range items {
			m, ok := item.(map[string]any)
			if !ok {
				t.Fatalf("oneOf[%d] is %T, want map[string]any", i, item)
			}
			out[i] = m
		}
		return out
	case []map[string]any:
		return items
	default:
		t.Fatalf("oneOf is %T, want a slice", raw)
		return nil
	}
}

func assertToolBranch(t *testing.T, schema map[string]any, name string, wantParams map[string]any) {
	t.Helper()
	for _, branch := range oneOfBranches(t, schema) {
		props, ok := branch["properties"].(map[string]any)
		if !ok {
			continue
		}
		tool, _ := props["tool"].(map[string]any)
		if tool["const"] != name {
			continue
		}
		args, _ := props["arguments"].(map[string]any)
		if wantParams == nil {
			if props["arguments"] != nil {
				t.Errorf("tool %q arguments = %#v, want nil Parameters passthrough", name, props["arguments"])
			}
			return
		}
		if !sameMap(args, wantParams) {
			t.Errorf("tool %q arguments were not passed through unmodified", name)
		}
		return
	}
	t.Fatalf("no oneOf branch for tool %q", name)
}

func toolArguments(t *testing.T, schema map[string]any, name string) map[string]any {
	t.Helper()
	for _, branch := range oneOfBranches(t, schema) {
		props, ok := branch["properties"].(map[string]any)
		if !ok {
			continue
		}
		tool, _ := props["tool"].(map[string]any)
		if tool["const"] != name {
			continue
		}
		args, _ := props["arguments"].(map[string]any)
		return args
	}
	t.Fatalf("no oneOf branch for tool %q", name)
	return nil
}

func assertTextBranch(t *testing.T, schema map[string]any) {
	t.Helper()
	for _, branch := range oneOfBranches(t, schema) {
		props, ok := branch["properties"].(map[string]any)
		if !ok {
			continue
		}
		text, ok := props["text"].(map[string]any)
		if !ok {
			continue
		}
		if text["type"] != "string" {
			t.Errorf("text branch type = %v, want string", text["type"])
		}
		return
	}
	t.Fatal("schema missing text-only oneOf branch")
}

func sameMap(a, b map[string]any) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return reflect.ValueOf(a).Pointer() == reflect.ValueOf(b).Pointer()
}
