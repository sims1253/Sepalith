// PRM05 source-window selection (policy `prm05-source-balanced-v1`), ported
// from packages/sepalith/src/sepalith/campaign_selection.py
// `select_source_window` without a scope pin: arm A's TRAIN rows all use
// scope_mode "off". The region is required; prefix and suffix lines are added
// alternately (prefix first, nearest first) until the next line on a side no
// longer fits the UTF-16 budget, which blocks that side.

export const SOURCE_SELECTION_POLICY_ID = "prm05-source-balanced-v1";
export const DEFAULT_SELECTION_UTF16_UNITS = 6000;

export interface SourceOmission { side: "prefix" | "suffix"; start_line: number; end_line: number }

export interface SourceSelection {
  budget_utf16_units: number;
  used_utf16_units: number;
  required_utf16_units: number;
  overflow: boolean;
  required_overflow: boolean;
  prefix: string[];
  region: string[];
  suffix: string[];
  omissions: SourceOmission[];
}

// JavaScript string length is the UTF-16 length; +1 for the line separator.
function lineCost(line: string): number { return line.length + 1; }

function appendOmission(omissions: SourceOmission[], side: "prefix" | "suffix", index: number): void {
  const last = omissions[omissions.length - 1];
  if (last && last.side === side && last.end_line + 1 === index) last.end_line = index;
  else omissions.push({ side, start_line: index, end_line: index });
}

export function selectSourceWindow(lines: readonly string[], regionStartLine: number, regionEndLine: number,
                                   maxUtf16Units = DEFAULT_SELECTION_UTF16_UNITS): SourceSelection {
  if (lines.some(line => line.includes("\n") || line.includes("\r"))) throw new Error("campaign_selection: lines must be LF-free, CR-free");
  if (!(Number.isSafeInteger(regionStartLine) && Number.isSafeInteger(regionEndLine)) || regionStartLine < 0
      || regionStartLine > regionEndLine || regionEndLine >= lines.length)
    throw new Error("campaign_selection: region lines are outside the source");
  const region = lines.slice(regionStartLine, regionEndLine + 1);
  const requiredUnits = region.reduce((n, line) => n + lineCost(line), 0);
  let remaining = Math.max(0, maxUtf16Units - requiredUnits);
  let used = 0;
  let prefixNext = regionStartLine - 1;
  let suffixNext = regionEndLine + 1;
  let prefixBlocked = regionStartLine === 0;
  let suffixBlocked = regionEndLine + 1 >= lines.length;
  const prefixSelected: number[] = [];
  const suffixSelected: number[] = [];
  let turn: "prefix" | "suffix" = "prefix";
  while (remaining > 0 && (!prefixBlocked || !suffixBlocked)) {
    let selected = false;
    const order: readonly ("prefix" | "suffix")[] = turn === "prefix" ? ["prefix", "suffix"] : ["suffix", "prefix"];
    for (const side of order) {
      if (side === "prefix") {
        if (prefixBlocked) continue;
        if (prefixNext < 0) { prefixBlocked = true; continue; }
        const cost = lineCost(lines[prefixNext]);
        if (cost > remaining) { prefixBlocked = true; continue; }
        prefixSelected.push(prefixNext);
        prefixNext--;
        remaining -= cost;
        used += cost;
      } else {
        if (suffixBlocked) continue;
        if (suffixNext >= lines.length) { suffixBlocked = true; continue; }
        const cost = lineCost(lines[suffixNext]);
        if (cost > remaining) { suffixBlocked = true; continue; }
        suffixSelected.push(suffixNext);
        suffixNext++;
        remaining -= cost;
        used += cost;
      }
      turn = side === "prefix" ? "suffix" : "prefix";
      selected = true;
      break;
    }
    if (!selected) break;
  }
  prefixSelected.sort((a, b) => a - b);
  const omissions: SourceOmission[] = [];
  const prefixSet = new Set(prefixSelected);
  const suffixSet = new Set(suffixSelected);
  for (let i = 0; i < regionStartLine; i++) if (!prefixSet.has(i)) appendOmission(omissions, "prefix", i);
  for (let i = regionEndLine + 1; i < lines.length; i++) if (!suffixSet.has(i)) appendOmission(omissions, "suffix", i);
  const usedUnits = requiredUnits + used;
  const requiredOverflow = requiredUnits > maxUtf16Units;
  return {
    budget_utf16_units: maxUtf16Units,
    used_utf16_units: usedUnits,
    required_utf16_units: requiredUnits,
    overflow: requiredOverflow || usedUnits > maxUtf16Units,
    required_overflow: requiredOverflow,
    prefix: prefixSelected.map(i => lines[i]),
    region,
    suffix: suffixSelected.map(i => lines[i]),
    omissions,
  };
}
