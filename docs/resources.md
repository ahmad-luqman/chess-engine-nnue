# Resources

## Primary references

- **Chess Programming Wiki** — the bible. https://www.chessprogramming.org
  - Perft results: https://www.chessprogramming.org/Perft_Results
  - Bitboards, Magic Bitboards, Alpha-Beta, NNUE, Transposition Table pages.
- **TalkChess forum** — where engine authors discuss. http://talkchess.com

## Tutorials / courses

- **Bitboard CHESS ENGINE in C** — Maksim Korzh ("Code Monkey King"), YouTube.
  From-scratch series; concepts translate directly to Rust.
- **VICE (Video Instructional Chess Engine)** — classic tutorial engine series.

## Engines to read (Rust, modern, clean)

- **Viridithas** — strong, well-documented, NNUE. Top learning target.
  https://github.com/cosmobobak/viridithas
- **Carp** — https://github.com/dede1751/carp
- **Akimbo** — small and readable. https://github.com/jw1912/akimbo
- **Svart** — https://github.com/crippa1337/svart

## Engines to read (C++ — advanced gold standard)

- **Stockfish** — https://github.com/official-stockfish/stockfish

## Protocols & specs

- **UCI protocol** spec (search "UCI protocol specification download").

## Testing & training tooling

- **Cute Chess / cutechess-cli** — https://github.com/cutechess/cutechess
- **fastchess** — https://github.com/Disservin/fastchess
- **OpenBench** — https://github.com/AndyGrant/OpenBench
- **SPRT** explainer — Chess Programming Wiki "Sequential Probability Ratio Test".
- **bullet** (NNUE trainer) — https://github.com/jw1912/bullet
- **nnue-pytorch** (reference) — https://github.com/official-stockfish/nnue-pytorch

## Data

- **Syzygy tablebases** — endgame perfection (≤7 pieces).
- **Standard opening sets** — UHO, Pohl, 8moves_v3 (balanced test openings).
