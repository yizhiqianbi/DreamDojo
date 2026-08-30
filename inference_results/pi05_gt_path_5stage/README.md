# Five-Stage GT-Anchored Parallel Futures

This artifact demonstrates a 240-step receding-horizon rollout along a recorded Jokeru path. Each of five stages contains seven π0.5 action samples rolled out by DreamDojo plus one hidden ground-truth candidate. The ground-truth slot uses both the recorded 48-step action sequence and the corresponding recorded 49-frame video.

- Stage 1: episode 0, control steps `100 → 148`; GT is display candidate 5.
- Stage 2: episode 0, control steps `148 → 196`; GT is display candidate 3.
- Stage 3: episode 0, control steps `196 → 244`; GT is display candidate 7.
- Stage 4: episode 0, control steps `244 → 292`; GT is display candidate 2.
- Stage 5: episode 0, control steps `292 → 340`; GT is display candidate 6.
- Selection: reveal the GT candidate after all eight futures play, promote its endpoint to the next observation, and sample the next set of futures from that recorded state.

`segment_02/` … `segment_05/` contain 28 newly sampled π0.5 action chunks and 28 newly generated DreamDojo videos. Stage 1 reuses seven branches from the original eight-way artifact. `segment_*_gt_*` contains each recorded action sequence converted both to π0.5's 30D space and DreamDojo's 384D condition. `manifest.json` records the hidden candidate order and the complete five-stage path.

The 46.5-second 1080p showcase is embedded in the [interactive viewer](https://dreamdojo-jokeru-lab.boingshaw.chatgpt.site) and stored at `../../viewer/public/videos/parallel8/parallel-universe-directors-cut.mp4`. Re-render it with `node scripts/render_parallel_universe_showcase.mjs`.

Rebuild the combined manifest with:

```bash
python integrations/pi05/build_gt_path_artifact.py
```
