/**
 * CLI. Three modes:
 *
 *   npm run play                          play black against the Romantic
 *   npm run play -- --persona solid --side black
 *   npm run play -- --self romantic:solid --moves 40
 *   npm run play -- --judge --fen "<fen>" --persona hustler
 */

import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { Chess } from "chess.js";
import { TypeSafeClient } from "@typesafe-ai/sdk";
import { PERSONAS, type Persona } from "./personas.js";
import { chooseMove, type Decision } from "./engine.js";

function arg(name: string, fallback?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] && !process.argv[i + 1].startsWith("--")
    ? process.argv[i + 1]
    : i >= 0
      ? ""
      : fallback;
}

function persona(key: string | undefined): Persona {
  const p = PERSONAS[key ?? "romantic"];
  if (!p) {
    console.error(`Unknown persona "${key}". Available: ${Object.keys(PERSONAS).join(", ")}`);
    process.exit(1);
  }
  return p;
}

function makeClient(): TypeSafeClient | null {
  if (!process.env.TYPESAFE_API_KEY) {
    console.log("TYPESAFE_API_KEY is not set. Running on the search alone, with no judgment layer.\n");
    return null;
  }
  return new TypeSafeClient({ timeout: 20000 });
}

function report(d: Decision, p: Persona) {
  const head = `${p.name} plays ${d.san}`;
  const meta = [
    `via ${d.source}`,
    d.confidence !== undefined ? `confidence ${d.confidence.toFixed(2)}` : null,
    d.underPressure !== undefined ? `pressure ${d.underPressure.toFixed(2)}` : null,
    d.usage ? `${d.usage.input_tokens + d.usage.output_tokens} tokens` : null,
  ].filter(Boolean).join(", ");

  console.log(`\n${head}  (${meta})`);
  console.log(`  ${d.note}`);

  if (d.table.length > 1) {
    console.log(`  ${"move".padEnd(8)}${"loss".padStart(5)}${"fit".padStart(7)}${"attack".padStart(8)}${"loose".padStart(7)}${"total".padStart(8)}`);
    for (const c of d.table) {
      const mark = c.san === d.san ? " <" : "";
      console.log(
        `  ${c.san.padEnd(8)}${String(c.loss).padStart(5)}` +
        `${c.fit.toFixed(2).padStart(7)}${c.aggression.toFixed(2).padStart(8)}` +
        `${c.looseness.toFixed(2).padStart(7)}${c.total.toFixed(3).padStart(8)}${mark}`,
      );
    }
  }
}

function outcome(chess: Chess): string | null {
  if (chess.isCheckmate()) return `Checkmate. ${chess.turn() === "w" ? "Black" : "White"} wins.`;
  if (chess.isStalemate()) return "Stalemate.";
  if (chess.isInsufficientMaterial()) return "Draw by insufficient material.";
  if (chess.isThreefoldRepetition()) return "Draw by repetition.";
  if (chess.isDraw()) return "Draw.";
  return null;
}

async function selfPlay(spec: string, maxMoves: number, client: TypeSafeClient | null) {
  const [wKey, bKey] = spec.split(":");
  const white = persona(wKey);
  const black = persona(bKey || wKey);
  const chess = new Chess(arg("fen") || undefined);

  console.log(`White: ${white.name}   Black: ${black.name}\n`);

  for (let i = 0; i < maxMoves * 2; i++) {
    const end = outcome(chess);
    if (end) { console.log(`\n${end}`); break; }

    const p = chess.turn() === "w" ? white : black;
    const d = await chooseMove(chess, p, client);
    chess.move(d.san);
    report(d, p);
  }

  console.log(`\n${chess.pgn()}`);
}

async function judgeOne(client: TypeSafeClient | null) {
  const chess = new Chess(arg("fen") || undefined);
  const p = persona(arg("persona"));
  console.log(chess.ascii());
  const d = await chooseMove(chess, p, client);
  report(d, p);
}

async function interactive(client: TypeSafeClient | null) {
  const p = persona(arg("persona"));
  const humanIsWhite = (arg("side") ?? "black") === "white";
  const chess = new Chess(arg("fen") || undefined);
  const rl = readline.createInterface({ input, output });

  console.log(`You are ${humanIsWhite ? "white" : "black"} against ${p.name}.`);
  console.log(`Enter moves in algebraic notation (e4, Nf3, O-O). "quit" to stop.\n`);

  while (true) {
    const end = outcome(chess);
    if (end) { console.log(`\n${end}`); break; }

    const humanToMove = (chess.turn() === "w") === humanIsWhite;

    if (humanToMove) {
      console.log(chess.ascii());
      const answer = (await rl.question("> ")).trim();
      if (answer === "quit" || answer === "") break;
      try {
        chess.move(answer);
      } catch {
        console.log(`Not a legal move. Options: ${chess.moves().slice(0, 12).join(", ")}...`);
      }
    } else {
      process.stdout.write("thinking...");
      const d = await chooseMove(chess, p, client);
      process.stdout.write("\r           \r");
      chess.move(d.san);
      report(d, p);
      console.log("");
    }
  }

  rl.close();
  console.log(`\n${chess.pgn()}`);
}

const client = makeClient();
const self = arg("self");

if (arg("judge") !== undefined) await judgeOne(client);
else if (self !== undefined) await selfPlay(self || "romantic:solid", Number(arg("moves") ?? 30), client);
else await interactive(client);
