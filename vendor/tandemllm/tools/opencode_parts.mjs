// What opencode shows for a recorded chat-completions stream (X1, 2026-10-03).
//
//   node tools/opencode_parts.mjs PROVIDER_ENTRY.mjs STREAM.sse > parts.json
//
// PROVIDER_ENTRY is opencode's own @ai-sdk/openai-compatible module, taken out of its binary by
// tools/opencode_bundle.py. The stream goes through that parser exactly as opencode reads it
// (createOpenAICompatible, chatModel, doStream, a fetch that answers with the recorded bytes);
// its stream parts then go through opencode's SessionProcessor rules, copied from 1.18.34:
//   text-start       a new text part (a fresh ascending id); text-delta appends to the current one
//   reasoning-start  a new reasoning part unless one with that id is open; reasoning-end closes it
//   tool-input-start a tool part
// Parts render in the order they were made. A reasoning part's title is the TUI's
// (`ma`): its text trimmed, when it starts with **title** and a blank line (or ends there).
// The next request's assistant message is opencode's (the provider's message converter): the
// text parts joined with nothing between them as content, the reasoning parts as
// reasoning_content.
// Prints {events, parts, message}.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [entry, sse] = process.argv.slice(2);
const { createOpenAICompatible } = await import(pathToFileURL(entry).href);
const bytes = readFileSync(sse);
const provider = createOpenAICompatible({
  name: "dgx",
  baseURL: "http://127.0.0.1:1/v1",
  fetch: async () => new Response(bytes, { status: 200, headers: { "content-type": "text/event-stream" } }),
});
const model = provider.chatModel("TandemLLM");
const { stream } = await model.doStream({ prompt: [{ role: "user", content: [{ type: "text", text: "go" }] }] });

const TITLE = /^\*\*([^*\n]+)\*\*(?:\r?\n\r?\n|$)/;
const events = [];
const parts = [];
const open = {};
let text = null;
for await (const p of stream) {
  events.push(p.type === "error" ? { type: "error", error: String(p.error) } : { type: p.type, id: p.id });
  if (p.type === "text-start") { text = { type: "text", text: "" }; parts.push(text); }
  else if (p.type === "text-delta") { if (text) text.text += p.delta; }
  else if (p.type === "text-end") { text = null; }
  else if (p.type === "reasoning-start") { if (!(p.id in open)) { open[p.id] = { type: "reasoning", text: "" }; parts.push(open[p.id]); } }
  else if (p.type === "reasoning-delta") { if (p.id in open) open[p.id].text += p.delta; }
  else if (p.type === "reasoning-end") { delete open[p.id]; }
  else if (p.type === "tool-input-start") { parts.push({ type: "tool", name: p.toolName, id: p.id }); }
}
for (const p of parts) {
  if (p.type !== "reasoning") continue;
  const m = p.text.trim().match(TITLE);
  p.title = m ? m[1].trim() : null;
}
const message = {
  role: "assistant",
  content: parts.filter((p) => p.type === "text").map((p) => p.text).join(""),
  reasoning_content: parts.filter((p) => p.type === "reasoning").map((p) => p.text).join(""),
};
process.stdout.write(JSON.stringify({ events, parts, message }));
