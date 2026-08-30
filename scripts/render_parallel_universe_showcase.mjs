#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const artifactDir = join(repoRoot, 'inference_results', 'pi05_gt_path_5stage');
const manifest = JSON.parse(readFileSync(join(artifactDir, 'manifest.json'), 'utf8'));
const outputPath = resolve(
  repoRoot,
  process.argv[2] ?? 'viewer/public/videos/parallel8/parallel-universe-directors-cut.mp4',
);

const WIDTH = 1920;
const HEIGHT = 1080;
const FPS = 30;
const ACTION_SECONDS = 1.6;
const FUTURE_SECONDS = 6.759;
const SELECT_SECONDS = 0.95;
const STAGE_SECONDS = ACTION_SECONDS + FUTURE_SECONDS + SELECT_SECONDS;
const DURATION = STAGE_SECONDS * manifest.segments.length;
const FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';
const FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';

const between = (start, end) => `enable='between(t,${start.toFixed(3)},${end.toFixed(3)})'`;
const escapeText = (value) => value.replaceAll('\\', '\\\\').replaceAll(':', '\\:').replaceAll("'", "\\'");
const drawText = ({ value, x, y, size, color = 'e8ecef', font = FONT, enable }) => {
  const parts = [
    `drawtext=fontfile=${font}`,
    `text='${escapeText(value)}'`,
    `x=${x}`,
    `y=${y}`,
    `fontsize=${size}`,
    `fontcolor=0x${color}`,
  ];
  if (enable) parts.push(enable);
  return parts.join(':');
};
const drawBox = ({ x, y, w, h, color, thickness = 'fill', enable }) => {
  const parts = [`drawbox=x=${x}`, `y=${y}`, `w=${w}`, `h=${h}`, `color=${color}`, `t=${thickness}`];
  if (enable) parts.push(enable);
  return parts.join(':');
};
const groupedTrace = (trace) => Array.from({ length: 12 }, (_, group) => {
  const values = trace.slice(group * 4, group * 4 + 4);
  return values.reduce((total, value) => total + value, 0) / values.length;
});

const filters = [`color=c=0x0a0d11:s=${WIDTH}x${HEIGHT}:r=${FPS}:d=${DURATION},format=yuv420p[base]`];
manifest.segments.forEach((segment, index) => {
  filters.push(
    `[${index}:v]scale=500:375:force_original_aspect_ratio=increase,crop=500:375,setsar=1,format=yuva420p[obs_${index}]`,
  );
});
let observationStage = 'base';
manifest.segments.forEach((_, index) => {
  const next = `obs_stage_${index}`;
  const start = index * STAGE_SECONDS;
  const end = start + STAGE_SECONDS;
  filters.push(`[${observationStage}][obs_${index}]overlay=x=58:y=190:enable='between(t,${start.toFixed(3)},${end.toFixed(3)})':eof_action=pass[${next}]`);
  observationStage = next;
});

const graphics = [];
const tilePositions = Array.from({ length: 8 }, (_, index) => ({
  x: 630 + (index % 4) * 310,
  y: 165 + Math.floor(index / 4) * 385,
}));

manifest.segments.forEach((segment, stageIndex) => {
  const stageStart = stageIndex * STAGE_SECONDS;
  const videoStart = stageStart + ACTION_SECONDS;
  const videoEnd = videoStart + FUTURE_SECONDS;
  const stageEnd = stageStart + STAGE_SECONDS;
  const stageEnable = between(stageStart, stageEnd);
  const actionEnable = between(stageStart, videoStart);
  const videoEnable = between(videoStart, stageEnd);

  graphics.push(drawText({ value: `OBSERVATION ${stageIndex + 1}`, x: 58, y: 91, size: 28, color: 'f0f2f4', font: FONT_BOLD, enable: stageEnable }));
  graphics.push(drawText({ value: `STEP ${segment.step} → ${segment.end_step}`, x: 58, y: 132, size: 16, color: '8a929a', font: FONT_BOLD, enable: stageEnable }));
  graphics.push(drawBox({ x: 55, y: 187, w: 506, h: 381, color: '0x6c747c@0.55', thickness: 2, enable: stageEnable }));
  graphics.push(drawText({ value: 'INSTRUCTION', x: 58, y: 620, size: 14, color: '8a929a', font: FONT_BOLD, enable: stageEnable }));
  graphics.push(drawText({ value: 'arrange the orange juice', x: 58, y: 650, size: 24, color: 'e8ecef', font: FONT_BOLD, enable: stageEnable }));
  graphics.push(drawText({ value: 'and green tea neatly', x: 58, y: 683, size: 24, color: 'e8ecef', font: FONT_BOLD, enable: stageEnable }));
  graphics.push(drawText({ value: 'GT PATH', x: 58, y: 782, size: 13, color: '8a929a', font: FONT_BOLD, enable: stageEnable }));
  for (let pathIndex = 0; pathIndex < manifest.segments.length; pathIndex += 1) {
    graphics.push(drawBox({
      x: 58 + pathIndex * 66,
      y: 818,
      w: 48,
      h: 5,
      color: pathIndex <= stageIndex ? '0xb8f36b@0.95' : '0x4b5158@0.70',
      enable: stageEnable,
    }));
  }
  graphics.push(drawText({ value: `${stageIndex + 1} / ${manifest.segments.length}`, x: 58, y: 844, size: 14, color: 'b8f36b', font: FONT_BOLD, enable: stageEnable }));

  graphics.push(drawText({ value: '8 FUTURE ACTIONS', x: 630, y: 88, size: 28, color: 'f0f2f4', font: FONT_BOLD, enable: actionEnable }));
  graphics.push(drawText({ value: '7 POLICY SAMPLES + 1 RECORDED ACTION · IDENTITY HIDDEN', x: 630, y: 126, size: 14, color: '8a929a', font: FONT_BOLD, enable: actionEnable }));
  graphics.push(drawText({ value: '8 FUTURE VIDEOS', x: 630, y: 88, size: 28, color: 'f0f2f4', font: FONT_BOLD, enable: videoEnable }));
  graphics.push(drawText({ value: 'PLAY ALL · REVEAL GT AFTER ROLLOUT', x: 630, y: 126, size: 14, color: '8a929a', font: FONT_BOLD, enable: videoEnable }));

  segment.candidates.forEach((candidate, candidateIndex) => {
    const { x, y } = tilePositions[candidateIndex];
    const groupTrace = groupedTrace(candidate.action_trace);
    graphics.push(drawBox({ x, y, w: 292, h: 292, color: '0x12171d@1', enable: actionEnable }));
    graphics.push(drawBox({ x, y, w: 292, h: 292, color: '0x5e6670@0.55', thickness: 2, enable: actionEnable }));
    graphics.push(drawText({ value: String(candidateIndex + 1).padStart(2, '0'), x: x + 18, y: y + 17, size: 18, color: 'dce1e5', font: FONT_BOLD, enable: actionEnable }));
    graphics.push(drawText({ value: '48 × 30D', x: x + 198, y: y + 21, size: 11, color: '747d86', font: FONT_BOLD, enable: actionEnable }));
    groupTrace.forEach((value, group) => {
      const height = 18 + Math.round((value / 100) * 130);
      graphics.push(drawBox({
        x: x + 19 + group * 21,
        y: y + 241 - height,
        w: 13,
        h: height,
        color: '0xc4cbd1@0.86',
        enable: actionEnable,
      }));
    });
    graphics.push(drawText({ value: '12 × 4-STEP GROUPS', x: x + 19, y: y + 261, size: 11, color: '747d86', font: FONT_BOLD, enable: actionEnable }));

    graphics.push(drawBox({ x, y, w: 292, h: 330, color: '0x10151a@1', enable: videoEnable }));
    graphics.push(drawBox({ x, y, w: 292, h: 330, color: '0x5e6670@0.48', thickness: 2, enable: videoEnable }));
  });
});
filters.push(`[${observationStage}]${graphics.join(',')}[graphics]`);

let inputIndex = manifest.segments.length;
const videoInputs = [];
manifest.segments.forEach((segment, stageIndex) => {
  const videoStart = stageIndex * STAGE_SECONDS + ACTION_SECONDS;
  segment.candidates.forEach((candidate, candidateIndex) => {
    const label = `future_${stageIndex}_${candidateIndex}`;
    filters.push(
      `[${inputIndex}:v]scale=284:213:force_original_aspect_ratio=increase,crop=284:213,setsar=1,tpad=stop_mode=clone:stop_duration=${SELECT_SECONDS + 0.5},setpts=PTS-STARTPTS+${videoStart.toFixed(3)}/TB[${label}]`,
    );
    videoInputs.push({ stageIndex, candidateIndex, label });
    inputIndex += 1;
  });
});

let videoStage = 'graphics';
videoInputs.forEach(({ stageIndex, candidateIndex, label }, index) => {
  const { x, y } = tilePositions[candidateIndex];
  const videoStart = stageIndex * STAGE_SECONDS + ACTION_SECONDS;
  const stageEnd = (stageIndex + 1) * STAGE_SECONDS;
  const next = `video_stage_${index}`;
  filters.push(`[${videoStage}][${label}]overlay=x=${x + 4}:y=${y + 42}:enable='between(t,${videoStart.toFixed(3)},${stageEnd.toFixed(3)})':eof_action=pass[${next}]`);
  videoStage = next;
});

const captions = [];
manifest.segments.forEach((segment, stageIndex) => {
  const videoStart = stageIndex * STAGE_SECONDS + ACTION_SECONDS;
  const videoEnd = videoStart + FUTURE_SECONDS;
  const stageEnd = (stageIndex + 1) * STAGE_SECONDS;
  const videoEnable = between(videoStart, stageEnd);
  const revealEnable = between(videoEnd, stageEnd);
  segment.candidates.forEach((candidate, candidateIndex) => {
    const { x, y } = tilePositions[candidateIndex];
    const groupTrace = groupedTrace(candidate.action_trace);
    captions.push(drawText({ value: String(candidateIndex + 1).padStart(2, '0'), x: x + 16, y: y + 14, size: 17, color: 'e1e5e8', font: FONT_BOLD, enable: videoEnable }));
    groupTrace.forEach((value, group) => {
      const height = 3 + Math.round((value / 100) * 27);
      captions.push(drawBox({
        x: x + 17 + group * 21,
        y: y + 316 - height,
        w: 13,
        h: height,
        color: '0xaeb6bd@0.80',
        enable: videoEnable,
      }));
    });
    if (candidateIndex !== segment.gt_slot) {
      captions.push(drawBox({ x, y, w: 292, h: 330, color: '0x080a0d@0.70', enable: revealEnable }));
    }
  });
  const selected = tilePositions[segment.gt_slot];
  captions.push(drawBox({ x: selected.x, y: selected.y, w: 292, h: 330, color: '0xb8f36b@1', thickness: 5, enable: revealEnable }));
  captions.push(drawBox({ x: selected.x + 181, y: selected.y + 8, w: 96, h: 26, color: '0xb8f36b@1', enable: revealEnable }));
  captions.push(drawText({ value: 'GT SELECTED', x: selected.x + 190, y: selected.y + 14, size: 12, color: '11160d', font: FONT_BOLD, enable: revealEnable }));
  captions.push(drawText({
    value: stageIndex + 1 < manifest.segments.length ? 'USE FINAL FRAME AS NEXT OBSERVATION' : '5-LAYER GT PATH COMPLETE',
    x: 630,
    y: 965,
    size: 17,
    color: 'b8f36b',
    font: FONT_BOLD,
    enable: revealEnable,
  }));
});
filters.push(`[${videoStage}]${captions.join(',')}[out]`);

const tempDir = mkdtempSync(join(tmpdir(), 'dreamdojo-gt-path-'));
const filterPath = join(tempDir, 'showcase.filter');
writeFileSync(filterPath, `${filters.join(';\n')}\n`, 'utf8');

const args = ['-y'];
manifest.segments.forEach((segment) => {
  args.push('-loop', '1', '-framerate', String(FPS), '-t', String(DURATION), '-i', resolve(artifactDir, segment.observation_image));
});
manifest.segments.forEach((segment) => {
  segment.candidates.forEach((candidate) => args.push('-i', resolve(artifactDir, candidate.video)));
});
args.push(
  '-filter_complex_script', filterPath,
  '-map', '[out]',
  '-an',
  '-c:v', 'libx264',
  '-preset', 'medium',
  '-crf', '21',
  '-pix_fmt', 'yuv420p',
  '-movflags', '+faststart',
  '-t', String(DURATION),
  outputPath,
);

console.log(`Rendering ${manifest.segments.length}-layer GT path to ${outputPath}`);
const result = spawnSync('ffmpeg', args, { cwd: repoRoot, stdio: 'inherit' });
rmSync(tempDir, { recursive: true, force: true });
if (result.status !== 0) process.exit(result.status ?? 1);
console.log('GT-path director cut complete.');
