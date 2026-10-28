/**
 * Child-only `bash` tool backed by Hive's rootless Docker sandbox.
 *
 * The child preset mounts this plugin instead of the host shell/sandbox stack.
 * Each call sends the child's workspace to `/v1/sandbox/exec`; the endpoint
 * validates that the path is a subdirectory of `/srv/hive-ai/workspaces`,
 * mounts it at `/workspace`, and runs the command in the pinned sandbox image
 * with no network, no host home, no Hive state, and a `hive.session` label.
 */

import { defineTool } from '@deepseek-ai/dsh-tools';

export const name = 'hive-docker-bash';
export const inject = ['tools', 'workingDirectory', 'systemPrompt'];

const DEFAULT_ENDPOINT = 'http://127.0.0.1:8765/v1/sandbox/exec';

/** Render the endpoint response as the text the child model sees. */
function renderResult(body) {
  const stdout = typeof body.stdout === 'string' ? body.stdout : '';
  const stderr = typeof body.stderr === 'string' ? body.stderr : '';
  let text = stdout;
  if (stderr.length > 0) {
    if (text.length > 0 && !text.endsWith('\n')) text += '\n';
    text += `[stderr]\n${stderr}`;
  }
  if (text.length === 0) text = '(no output)';
  const markers = [];
  if (body.timed_out === true) markers.push('[timed out]');
  if (typeof body.exit_code === 'number' && body.exit_code !== 0) {
    markers.push(`[exit code: ${body.exit_code}]`);
  }
  if (markers.length > 0) {
    if (!text.endsWith('\n')) text += '\n';
    text += markers.join('\n');
  }
  return text;
}

export function apply(ctx, config = {}) {
  const endpoint = String(config.endpoint || DEFAULT_ENDPOINT);
  ctx.systemPrompt.section({
    name: 'tool:hive-docker-bash',
    order: 1000,
    text: 'Check the [exit code: N] marker on every bash result; the sandbox has no network and only /workspace is writable.',
  });
  ctx.tools.register(defineTool({
    name: 'bash',
    description: 'Execute a POSIX shell command in an isolated Docker container. The workspace is mounted at /workspace and is the only writable host path. The container has no network, no host home, and no Hive state.',
    parameters: {
      command: {
        type: 'string',
        required: true,
        description: 'The POSIX shell command to execute in /workspace.',
      },
      description: {
        type: 'string',
        description: 'Clear, concise description of what this command does in active voice, 5-10 words.',
      },
    },
    output: {
      schema: {
        type: 'object',
        properties: { text: { type: 'string' } },
        required: ['text'],
        additionalProperties: false,
      },
      render(_args, value) {
        return [{ type: 'text', text: value.text }];
      },
    },
    timeoutMs: 90000,
    async execute(args, exec) {
      if (exec.agent === undefined) {
        throw new Error('hive-docker-bash requires an agent session');
      }
      const workspace = await ctx.workingDirectory.ensure(exec.agent, exec.signal);
      const session = exec.agent.session;
      const header = session.header;
      const parent = header.parentSession ?? header.origin?.parentSession ?? '';
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          command: args.command,
          session_id: header.id,
          parent_session_id: parent,
          workspace,
          timeout: 60,
          use_docker: true,
          allow_local_fallback: false,
        }),
        signal: exec.signal,
      });
      let body = {};
      try {
        body = await response.json();
      } catch {
        body = {};
      }
      if (!response.ok) {
        let detail = '';
        try {
          detail = JSON.stringify(body).slice(0, 800);
        } catch {
          detail = '';
        }
        throw new Error(`docker sandbox request failed: HTTP ${response.status} ${detail}`);
      }
      return { text: renderResult(body) };
    },
  }));
}
