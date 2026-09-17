// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"context"
	"fmt"
	"time"
)

// hostAgentCLIMaxOutput caps how much of the CLI stdout we buffer. A review
// response is large but bounded; refusing further writes makes the child die
// of SIGPIPE instead of growing the heap. Package var so tests can shrink it.
var hostAgentCLIMaxOutput = 16 << 20

// hostAgentCLIWaitDelay bounds how long Wait keeps waiting on the child's
// stdout pipe after ctx cancellation. Package var so tests can shrink it.
var hostAgentCLIWaitDelay = 5 * time.Second

type cliTransport struct {
	command   string
	extraArgs []string
	extraEnv  []string
}

func newCLITransport(command string, extraArgs []string) *cliTransport {
	return &cliTransport{command: command, extraArgs: extraArgs}
}

func (t *cliTransport) Complete(ctx context.Context, req HostAgentRequest) ([]byte, *UsageInfo, error) {
	return nil, nil, fmt.Errorf("not implemented")
}
