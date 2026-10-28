/**
 * Synchronous, waiting delegation tool for the Hive parent.
 *
 * The stock DSH subagent tool returns a child id immediately and delivers the
 * result later as a completion notice.  The local parent model has no wait
 * primitive, so it reaches for job_output and sometimes starts duplicate
 * children.  This tool starts the same sandboxed child preset with
 * `delivery: 'caller'`, awaits the activation result, returns the child's final
 * text as the tool result, and disposes the activation.
 */

import { defineTool } from '@deepseek-ai/dsh-tools';

export const name = 'hive-delegate';
export const inject = ['tools', 'subagents', 'agentPresets', 'systemPrompt'];

function renderResult(text) {
  return [{ type: 'text', text }];
}

export async function apply(ctx, config = {}) {
  const providerName = String(config.provider || 'spawn');
  const agentPreset = String(config.agentPreset || '').trim();
  const toolName = String(config.toolName || 'delegate');
  if (!agentPreset) throw new Error('hive-delegate: agentPreset is required');
  const presets = ctx.get('agentPresets');
  if (presets === undefined) {
    throw new Error('hive-delegate: agent preset selection requires the agent-presets service');
  }
  await presets.readDocument(agentPreset);
  const provider = ctx.subagents.getProvider(providerName);
  if (provider === undefined) {
    throw new Error(`hive-delegate: provider "${providerName}" is not registered`);
  }
  if (!provider.capabilities.agentPreset) {
    throw new Error(`hive-delegate: provider "${providerName}" does not support an independent agent preset`);
  }
  if (!provider.capabilities.depthLimit) {
    throw new Error(`hive-delegate: provider "${providerName}" does not support maxDepth`);
  }
  const maxDepth = Number.isSafeInteger(config.maxDepth) ? config.maxDepth : 1;

  ctx.systemPrompt.section({
    name: `tool:${toolName}`,
    order: 1000,
    text: `Use \`${toolName}\` for a self-contained sandboxed task: it waits for the child and returns the child's final answer as the tool result. Do not poll job_output or job_list for a child, and do not start the same task twice.`,
  });

  ctx.tools.register(defineTool({
    name: toolName,
    description: 'Delegate a self-contained task to a sandboxed child and wait for its final answer. The child runs in a Docker sandbox with only its workspace mounted at /workspace, no network, no host home, and no Hive state. The child result is returned directly by this call; do not poll job_output.',
    parameters: {
      cwd: {
        type: 'string',
        description: 'Existing scratch directory under /srv/hive-ai/workspaces. The child workspace is mounted at /workspace.',
      },
      description: {
        type: 'string',
        description: 'Short display label for the delegated task.',
      },
      prompt: {
        type: 'string',
        required: true,
        description: 'The complete, self-contained task for the child. It does not share this conversation, so include everything it needs.',
      },
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          text: { type: 'string', required: true },
        },
        additionalProperties: false,
      },
      render(_args, value) {
        return renderResult(value.text);
      },
    },
    timeoutMs: 600000,
    async execute(args, exec) {
      const parent = exec.agent;
      if (parent === undefined) {
        throw new Error('hive-delegate requires a calling agent');
      }
      const activation = await ctx.subagents.startActivation({
        provider: providerName,
        label: String(args.description || 'delegated task'),
        request: {
          prompt: [{ type: 'text', text: args.prompt }],
          parent,
          agentPreset,
          maxDepth,
          ...(args.cwd === undefined ? {} : { cwd: args.cwd }),
        },
        signal: exec.signal,
        delivery: 'caller',
      });
      let result;
      try {
        result = await activation.result;
      } finally {
        try {
          await activation.dispose();
        } catch {
          // The child result is already captured; teardown failure must not
          // replace it with an infrastructure error.
        }
      }
      const text = result.output
        .filter((block) => block && block.type === 'text')
        .map((block) => block.text)
        .join('\n')
        .trim();
      if (result.stopReason !== 'completed') {
        const detail = String(result.diagnostic || '').trim();
        return {
          text: `[child ${result.stopReason}]${detail ? ` ${detail}` : ''}\n${text || '(no output)'}`,
        };
      }
      return { text: text || '(child produced no output)' };
    },
  }));
}
