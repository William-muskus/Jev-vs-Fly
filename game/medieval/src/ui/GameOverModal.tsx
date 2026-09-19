import { useEffect, useRef, useState } from "react";
import { Check, Copy, Home, PauseCircle, RotateCw, Swords } from "lucide-react";

import { DEMO_REMATCH_DELAY_MS } from "../core/gameController";
import type { Difficulty, EndReason, Faction, GameResult } from "../core/types";
import { Crest } from "./Heraldry";

/**
 * Extra framing for an AI vs AI duel. The viewer is an audience,
 * not a combatant: the dialog names the two engines, counts the loop down where
 * one is queued, and offers to hold the board so the final position can be read.
 */
export interface ShowcaseOutcome {
  /** 1-based duel counter within this AI vs AI session. */
  round: number;
  white: Difficulty;
  black: Difficulty;
  /** The loop is armed — another duel starts on its own. */
  autoRematch: boolean;
  /** Live ms left on the queued rematch, or null when nothing is queued. */
  getRematchRemaining: () => number | null;
  /** Disarm the loop so the board stays on the final position. */
  onHold: () => void;
}

interface GameOverModalProps {
  result: GameResult;
  pgn: string;
  playerColor: Faction;
  versusComputer: boolean;
  moveCount: number;
  showcase?: ShowcaseOutcome | null;
  onRematch: () => void;
  onMenu: () => void;
}

const REASON_COPY: Record<EndReason, string> = {
  checkmate: "Checkmate",
  stalemate: "Stalemate — the king cannot move",
  resignation: "The banner was lowered",
  timeout: "The hourglass ran dry",
  threefold: "Threefold repetition",
  insufficient: "Insufficient material",
  fiftymove: "Fifty-move rule",
  draw: "Drawn position",
};

const ENGINE_NAME: Record<Difficulty, string> = {
  easy: "Squire",
  medium: "Knight",
  hard: "Warlord",
};

export function GameOverModal({
  result,
  pgn,
  playerColor,
  versusComputer,
  moveCount,
  showcase,
  onRematch,
  onMenu,
}: GameOverModalProps) {
  const [copied, setCopied] = useState(false);

  const draw = result.winner === null;
  const playerWon = versusComputer && result.winner === playerColor;
  const headline = draw
    ? "A DRAW"
    : showcase
      ? result.winner === "w"
        ? "IVORY TRIUMPHS"
        : "OBSIDIAN TRIUMPHS"
      : playerWon
        ? "VICTORY"
        : versusComputer
          ? "DEFEAT"
          : result.winner === "w"
            ? "IVORY TRIUMPHS"
            : "OBSIDIAN TRIUMPHS";

  const copyPgn = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(pgn);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.warn("[ui] clipboard unavailable", error);
    }
  };

  // A showcase is watched, so the final position is the picture: the backdrop
  // stays thin and unblurred there instead of hiding the hall behind the card.
  return (
    <div
      className={`mc-modal-pad pointer-events-auto absolute inset-0 z-30 flex items-center justify-center ${
        showcase ? "bg-black/35" : "bg-black/65 backdrop-blur-[3px]"
      }`}
    >
      <div className="mc-parchment mc-goldleaf mc-rise w-full max-w-md overflow-hidden">
        <div className="px-6 pb-6 pt-7 text-center">
          {showcase ? (
            <p className="mc-display text-[0.55rem] tracking-[0.42em] text-[#8a6b3a]">
              AI VS AI · DUEL {showcase.round}
            </p>
          ) : null}

          <div className={`flex justify-center gap-3 ${showcase ? "mt-3" : ""}`}>
            {result.winner ? (
              <Crest faction={result.winner} size={44} active />
            ) : (
              <>
                <Crest faction="w" size={34} />
                <Crest faction="b" size={34} />
              </>
            )}
          </div>

          <h2 className="mc-display mt-4 text-3xl font-bold tracking-[0.14em] text-[#43301a]">{headline}</h2>
          <div className="mc-rule mx-auto mt-2 w-40 opacity-70" />
          <p className="mt-2 text-sm italic text-[#6a5334]">{REASON_COPY[result.reason]}</p>

          {showcase ? (
            <p className="mc-display mt-2 text-[0.6rem] tracking-[0.24em] text-[#7d6236]">
              {ENGINE_NAME[showcase.white]} <span className="text-[#a2854c]">vs</span> {ENGINE_NAME[showcase.black]} ·{" "}
              {moveCount} {moveCount === 1 ? "move" : "moves"}
            </p>
          ) : null}

          <div className="mt-5 max-h-24 overflow-y-auto rounded-sm border border-[#8a652255] bg-[#00000010] p-3 text-left font-mono text-[0.7rem] leading-relaxed text-[#4a3a24]">
            {pgn.length > 0 ? pgn : "1. (no moves)"}
          </div>

          {showcase?.autoRematch ? (
            <NextDuelCountdown getRemaining={showcase.getRematchRemaining} onHold={showcase.onHold} />
          ) : null}

          <div className="mt-5 grid grid-cols-2 gap-2">
            <button type="button" className="mc-btn mc-btn-primary flex items-center justify-center gap-2" onClick={onRematch}>
              {showcase ? <RotateCw size={15} /> : <Swords size={15} />} {showcase ? "Another duel" : "Rematch"}
            </button>
            <button type="button" className="mc-btn flex items-center justify-center gap-2" onClick={onMenu}>
              <Home size={15} /> Great hall
            </button>
          </div>
          <button
            type="button"
            className="mc-btn mt-2 flex w-full items-center justify-center gap-2"
            onClick={() => void copyPgn()}
          >
            {copied ? <Check size={15} /> : <Copy size={15} />} {copied ? "Copied" : "Copy record (PGN)"}
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * The showcase loop, made visible. It reads the deadline off the controller on
 * its own 100 ms tick — the countdown must not push a snapshot through the whole
 * interface once every tenth of a second — and offers the one control the loop
 * was missing: a way to stop it without hunting for the transport rail.
 */
function NextDuelCountdown({ getRemaining, onHold }: { getRemaining: () => number | null; onHold: () => void }) {
  const [remaining, setRemaining] = useState<number | null>(() => getRemaining());
  const read = useRef(getRemaining);
  read.current = getRemaining;

  useEffect(() => {
    const timer = setInterval(() => setRemaining(read.current()), 100);
    return () => clearInterval(timer);
  }, []);

  if (remaining === null) return null;
  const seconds = Math.max(1, Math.ceil(remaining / 1000));
  const ratio = Math.min(1, Math.max(0, remaining / DEMO_REMATCH_DELAY_MS));

  return (
    <div className="mt-4 rounded-sm border border-[#8a652244] bg-[#00000010] px-3 py-2.5">
      <div className="flex items-center justify-between gap-3">
        <p className="mc-display text-[0.58rem] tracking-[0.26em] text-[#7d6236]">NEXT DUEL IN {seconds}s</p>
        <button
          type="button"
          className="mc-display flex items-center gap-1 rounded-sm border border-[#8a652277] px-2 py-1 text-[0.55rem] tracking-[0.2em] text-[#5f4a26] transition-colors hover:border-[#8a6522] hover:text-[#3d2d15]"
          onClick={onHold}
        >
          <PauseCircle size={12} /> HOLD
        </button>
      </div>
      <div className="mt-2 h-[3px] overflow-hidden rounded-full bg-[#00000022]">
        <div
          className="h-full rounded-full bg-gradient-to-r from-[#8a6522] via-[#d8b163] to-[#8a6522] transition-[width] duration-100 ease-linear"
          style={{ width: `${ratio * 100}%` }}
        />
      </div>
    </div>
  );
}
