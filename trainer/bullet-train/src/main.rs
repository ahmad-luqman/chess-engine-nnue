//! bullet trainer for chess-engine-nnue's first NNUE (issue #45).
//!
//! Architecture: `(768 -> 256)x2 -> 1x8`, SCReLU, `Chess768` perspective inputs
//! (NOT HalfKP — king-agnostic, see ADR 0016 and `docs/04-nnue.md`).
//!
//! Quantisation: `QA=255`, `QB=64`, `SCALE=400`. Output buckets:
//! `MaterialCount<8>`, i.e. `bucket = (piece_count - 2) / 4`.
//!
//! The exported `quantised.bin` byte layout is the inference contract consumed by
//! engine wiring (#46): see `docs/decisions/0016-nnue-first-net-architecture.md`.
//! The `verify` crate is a CPU reference implementation of that contract.
//!
//! Knobs are environment variables so the one binary serves both a short
//! proof-of-life run and the full schedule (defaults = full schedule):
//!   NNUE_DATA          path to a bullet-readable Stockfish/Leela binpack
//!   NNUE_NET_ID        checkpoint name (default chess-engine-nnue-0001)
//!   NNUE_SUPERBATCHES  end superbatch (default 40)
//!   NNUE_BATCHES_PER_SB batches per superbatch (default 6104 ~= 100M positions)
//!   NNUE_THREADS       data-loader threads (default 4)
//!   NNUE_SAVE_RATE     checkpoint every N superbatches (default 10)

use bullet_lib::{
    game::{inputs::Chess768, outputs::MaterialCount},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{
        loader::sfbinpack::{MoveType, PieceType, SfBinpackLoader, TrainingDataEntry},
        ValueTrainerBuilder,
    },
};

const HL_SIZE: usize = 256;
const NUM_OUTPUT_BUCKETS: usize = 8;
const QA: i16 = 255;
const QB: i16 = 64;
const SCALE: i32 = 400;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let dataset =
        std::env::var("NNUE_DATA").unwrap_or_else(|_| "data/bootstrap.binpack".to_string());
    let net_id =
        std::env::var("NNUE_NET_ID").unwrap_or_else(|_| "chess-engine-nnue-0001".to_string());
    let superbatches = env_usize("NNUE_SUPERBATCHES", 40);
    let batches_per_superbatch = env_usize("NNUE_BATCHES_PER_SB", 6104);
    let threads = env_usize("NNUE_THREADS", 4);
    let save_rate = env_usize("NNUE_SAVE_RATE", 10);

    let mut trainer = ValueTrainerBuilder::default()
        // two accumulators: side-to-move and opponent (the "x2").
        .dual_perspective()
        .optimiser(AdamW)
        // plain 768 piece-square inputs.
        .inputs(Chess768)
        // 8 output buckets selected by piece count: (occ.count_ones() - 2) / 4.
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        // Save layout = the #46 inference contract. Order and quantisation matter:
        //   l0w (QA), l0b (QA), l1w (QB, transposed for per-bucket contiguity),
        //   l1b (QA*QB). All column-major, little-endian i16.
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        // labels are in [0, 1]; map the net output through a sigmoid to match.
        // target = wdl * game_result + (1 - wdl) * sigmoid(cp_score / SCALE).
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768, HL_SIZE);
            let l1 = builder.new_affine("l1", 2 * HL_SIZE, NUM_OUTPUT_BUCKETS);

            // stm accumulator FIRST, opponent second (the concat order #46 must
            // replicate; getting this backwards silently costs hundreds of Elo).
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 18 },
        save_rate,
    };

    let settings =
        LocalSettings { threads, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    // Standard Stockfish-binpack filter: skip the opening, in-check, extreme-score,
    // and tactical (capture/promo) positions — they make poor eval targets.
    fn filter(entry: &TrainingDataEntry) -> bool {
        entry.ply >= 16
            && !entry.pos.is_checked(entry.pos.side_to_move())
            && entry.score.unsigned_abs() <= 10000
            && entry.mv.mtype() == MoveType::Normal
            && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
    }

    let data_loader = SfBinpackLoader::new(&dataset, 1024, threads, filter);

    trainer.run(&schedule, &settings, &data_loader);
}
