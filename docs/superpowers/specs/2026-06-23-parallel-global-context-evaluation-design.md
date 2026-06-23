# Parallel Global-Context Evaluation Design

## Goal

Reduce the remaining genuine incremental global-context D evaluation from
several days to roughly one day without changing model scores, beam-search
semantics, calibration rules, or existing per-piece checkpoints.

## Execution model

Four independent worker processes load the same frozen models and dataset. Each
worker owns a deterministic slice of piece IDs using the sorted piece list:
`piece_ids[worker_index::worker_count]`. Workers may traverse the same
configuration grid, but no two workers compute or write the same piece.

The existing checkpoint identity remains unchanged:

`progress/<split>/<config>/<piece_id>.json`

Existing checkpoints are reused regardless of which worker created them.
Writes use a temporary file followed by an atomic replace so interruption
cannot leave a partially written JSON checkpoint.

## Phases

- `calibration`: workers compute only their assigned calibration pieces for all
  12 configurations.
- `finalize`: a single process verifies that every calibration checkpoint
  exists, aggregates the grid, selects the best configuration, and stores it in
  the progress directory. If all matching test checkpoints also exist, it
  writes the final result JSON.
- `test`: workers read the stored selected configuration and compute only their
  assigned test pieces.

The launcher runs four calibration workers, waits, runs `finalize` to select the
configuration, runs four test workers, waits, and runs `finalize` again.

## Safety and compatibility

- Existing 118+ checkpoints are retained and skipped.
- Worker arguments are validated: count must be positive and index must be in
  `[0, worker_count)`.
- Finalization fails clearly when calibration is incomplete and never selects
  from a partial grid.
- Test workers refuse to start before a selected calibration file exists.
- Metrics are aggregated from all expected piece checkpoint files, independent
  of worker ownership.
- A one-worker run remains equivalent to the previous serial behavior.

## Verification

Unit tests cover deterministic disjoint partitioning, complete coverage,
incomplete-grid rejection, aggregation across all workers, selected-config
persistence, and atomic checkpoint writing. Remote smoke verification runs
multiple workers on already completed checkpoints before the long process is
stopped.
