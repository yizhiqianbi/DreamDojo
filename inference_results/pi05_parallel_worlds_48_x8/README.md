# π0.5 → DreamDojo: 8 Parallel 48-Step Futures

This artifact starts from the same recorded Jokeru observation at episode `0`, step `100` (source frame `400`). π0.5 samples eight independent 48-step, 30D action plans in one warm model process. Each plan is projected to DreamDojo's 384D action condition and rolled through four chained 12-action windows.

## Outputs

- `universe_00.mp4` … `universe_07.mp4`: eight generated 49-frame futures.
- `observation_replay.mp4`: the recorded observation stream used by the mock VLA loop; it is not generated.
- `universe_*_pi05_actions.npy`: π0.5 actions with shape `[48, 30]`.
- `universe_*_dreamdojo_actions.npy`: DreamDojo conditions with shape `[48, 384]`.
- `manifest.json`: seeds, VLA timing, chunk boundaries, and all 28 pairwise video-difference measurements.
- `final_frame_contact_sheet.png`: a visual QA grid of the eight final generated frames.
- `../../viewer/public/videos/parallel8/parallel-universe-directors-cut.mp4`: a 46.5-second 1080p showcase built from the five-stage GT-anchored artifact.

## Director's cut

The current showcase continues this first stage through four more recorded endpoints. At every stage it presents eight action candidates, plays all eight futures, reveals the hidden GT candidate, and uses that candidate's final frame as the next observation. See [`../pi05_gt_path_5stage`](../pi05_gt_path_5stage). It can be reproduced with `node scripts/render_parallel_universe_showcase.mjs`.

All nine videos were decoded and verified as `640×480`, 49 frames, and `7.25 FPS`. Action RMS distance from Universe A ranges from `0.0778` to `0.1094`; final video PSNR against Universe A ranges from `16.40 dB` to `18.24 dB`, confirming that distinct policy samples produce visibly different generated futures.

These videos use the 35-step DreamDojo teacher. The Jokeru Self-Forcing training path is integrated separately under `integrations/self_forcing/`; no accelerated-student result is claimed until its student checkpoint is trained.
