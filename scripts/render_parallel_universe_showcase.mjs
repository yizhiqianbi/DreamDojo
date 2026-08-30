#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const artifactDir = join(repoRoot, 'inference_results', 'pi05_parallel_worlds_48_x8');
const manifestPath = join(repoRoot, 'viewer', 'public', 'data', 'pi05-parallel-x8-manifest.json');
const outputPath = resolve(
  repoRoot,
  process.argv[2] ?? 'viewer/public/videos/parallel8/parallel-universe-directors-cut.mp4',
);
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));

const WIDTH = 1920;
const HEIGHT = 1080;
const FPS = 30;
const VIDEO_START = 5.8;
const VIDEO_END = 12.56;
const DURATION = 15.2;
const FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';
const FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';
const COLORS = ['a3ff52', '52d9ff', 'ffbf5b', 'c68cff', 'ff6f91', '5ff0c8', 'f7e75b', '719cff'];

const between = (start, end) => `enable='between(t,${start},${end})'`;
const text = (value) => value.replaceAll('\\', '\\\\').replaceAll(':', '\\:').replaceAll("'", "\\'");
const drawText = ({ value, x, y, size, color = 'ffffff', font = FONT, enable, alpha, spacing }) => {
  const parts = [
    `drawtext=fontfile=${font}`,
    `text='${text(value)}'`,
    `x=${x}`,
    `y=${y}`,
    `fontsize=${size}`,
    `fontcolor=0x${color}`,
  ];
  if (enable) parts.push(enable);
  if (alpha) parts.push(`alpha='${alpha}'`);
  if (spacing) parts.push(`line_spacing=${spacing}`);
  return parts.join(':');
};
const drawBox = ({ x, y, w, h, color, thickness = 'fill', enable }) => {
  const parts = [`drawbox=x=${x}`, `y=${y}`, `w=${w}`, `h=${h}`, `color=${color}`, `t=${thickness}`];
  if (enable) parts.push(enable);
  return parts.join(':');
};

const filters = [];
filters.push(`color=c=0x07111d:s=${WIDTH}x${HEIGHT}:r=${FPS}:d=${DURATION},format=yuv420p[base]`);
filters.push(
  `[0:v]scale=560:420:force_original_aspect_ratio=increase,crop=560:420,setsar=1,format=yuva420p,fade=t=in:st=0.25:d=0.6:alpha=1[observation]`,
);
filters.push('[base][observation]overlay=x=70:y=248:eof_action=pass[stage]');

const graphics = [];

// Persistent observation column.
graphics.push(drawBox({ x: 56, y: 166, w: 588, h: 700, color: '0x0d1927@0.96', thickness: 2 }));
graphics.push(drawBox({ x: 70, y: 236, w: 560, h: 444, color: '0xa3ff52@0.42', thickness: 2 }));
graphics.push(drawBox({ x: 70, y: 184, w: 76, h: 5, color: '0xa3ff52@0.95' }));
graphics.push(drawText({ value: 'OBSERVATION / T₀', x: 70, y: 199, size: 18, color: '8ca1b3', font: FONT_BOLD }));
graphics.push(drawText({ value: 'RECORDED JOKERU FRAME', x: 70, y: 220, size: 30, color: 'f2f8f6', font: FONT_BOLD }));
graphics.push(drawBox({ x: 86, y: 624, w: 188, h: 31, color: '0x05090f@0.78' }));
graphics.push(drawText({ value: 'FRAME 400 · 7.25 HZ', x: 99, y: 631, size: 14, color: 'a3ff52', font: FONT_BOLD }));
graphics.push(drawBox({ x: 70, y: 698, w: 560, h: 144, color: '0x07101a@0.96', thickness: 2 }));
graphics.push(drawText({ value: 'INSTRUCTION', x: 92, y: 718, size: 16, color: '8ca1b3', font: FONT_BOLD }));
graphics.push(drawText({ value: '“arrange the orange juice', x: 92, y: 749, size: 29, color: 'f2f8f6', font: FONT_BOLD }));
graphics.push(drawText({ value: '& green tea neatly”', x: 92, y: 786, size: 29, color: 'f2f8f6', font: FONT_BOLD }));
graphics.push(drawText({ value: 'SAME OBSERVATION FOR EVERY BRANCH', x: 70, y: 888, size: 14, color: '6f8598', font: FONT_BOLD }));

// Global title and intro beat.
graphics.push(drawText({ value: 'DREAMDOJO × π0.5', x: 70, y: 54, size: 21, color: 'a3ff52', font: FONT_BOLD }));
graphics.push(drawText({ value: 'PARALLEL UNIVERSE ROLLOUT', x: 70, y: 84, size: 15, color: '71889b', font: FONT_BOLD }));
graphics.push(drawText({ value: 'FUTURE IS', x: 742, y: 296, size: 88, color: 'f1f7f5', font: FONT_BOLD, enable: between(0.45, 2.3) }));
graphics.push(drawText({ value: 'NOT SINGULAR.', x: 742, y: 386, size: 88, color: 'a3ff52', font: FONT_BOLD, enable: between(0.65, 2.3) }));
graphics.push(drawText({ value: 'ONE OBSERVATION  /  EIGHT STOCHASTIC POLICY SAMPLES', x: 748, y: 505, size: 20, color: '8298aa', font: FONT_BOLD, enable: between(0.9, 2.3) }));
COLORS.forEach((color, index) => {
  graphics.push(drawBox({ x: 748 + index * 96, y: 562, w: 72, h: 5, color: `0x${color}@0.95`, enable: between(1.05 + index * 0.06, 2.3) }));
});

// Action-plan beat: eight real 48-step magnitude traces reveal from left to right.
graphics.push(drawText({ value: '8 FUTURE ACTIONS', x: 780, y: 94, size: 43, color: 'f2f8f6', font: FONT_BOLD, enable: between(2.25, VIDEO_START) }));
graphics.push(drawText({ value: 'π0.5 FLOW MATCHING · 48 STEPS × 30D PER BRANCH', x: 782, y: 145, size: 17, color: '8298aa', font: FONT_BOLD, enable: between(2.25, VIDEO_START) }));
graphics.push(drawBox({ x: 652, y: 520, w: 92, h: 2, color: '0xa3ff52@0.55', enable: between(2.25, VIDEO_START) }));
graphics.push(drawBox({ x: 742, y: 223, w: 2, h: 672, color: '0xa3ff52@0.32', enable: between(2.25, VIDEO_START) }));

manifest.universes.slice(0, 8).forEach((universe, universeIndex) => {
  const y = 206 + universeIndex * 88;
  const color = COLORS[universeIndex];
  const reveal = 2.34 + universeIndex * 0.08;
  graphics.push(drawBox({ x: 780, y, w: 1065, h: 74, color: '0x101e2c@0.96', enable: between(reveal, VIDEO_START) }));
  graphics.push(drawBox({ x: 780, y, w: 1065, h: 74, color: `0x${color}@0.50`, thickness: 2, enable: between(reveal, VIDEO_START) }));
  graphics.push(drawBox({ x: 744, y: y + 36, w: 36, h: 2, color: `0x${color}@0.70`, enable: between(reveal, VIDEO_START) }));
  graphics.push(drawText({ value: `U${String(universeIndex + 1).padStart(2, '0')}`, x: 800, y: y + 18, size: 19, color, font: FONT_BOLD, enable: between(reveal, VIDEO_START) }));
  graphics.push(drawText({ value: `SEED ${universe.seed}`, x: 800, y: y + 44, size: 12, color: '758b9d', font: FONT_BOLD, enable: between(reveal, VIDEO_START) }));
  const trace = universe.action_trace.slice(0, 48);
  trace.forEach((value, step) => {
    const barHeight = 7 + Math.round((value / 100) * 42);
    const barX = 946 + step * 17;
    const barY = y + 60 - barHeight;
    const stepReveal = reveal + 0.22 + step * 0.045;
    graphics.push(drawBox({ x: barX, y: barY, w: 10, h: barHeight, color: `0x${color}@0.88`, enable: between(stepReveal.toFixed(3), VIDEO_START) }));
  });
  graphics.push(drawText({ value: `Δ ${universe.action_rms_from_universe_0.toFixed(3)}`, x: 1777, y: y + 26, size: 13, color: '91a5b5', font: FONT_BOLD, enable: between(reveal, VIDEO_START) }));
});

// Frame backgrounds for the 4×2 synchronized video wall.
const tilePositions = manifest.universes.slice(0, 8).map((_, index) => ({
  x: 686 + (index % 4) * 291,
  y: 205 + Math.floor(index / 4) * 386,
}));
graphics.push(drawText({ value: '8 GENERATED FUTURES', x: 686, y: 87, size: 43, color: 'f2f8f6', font: FONT_BOLD, enable: `enable='gte(t,5.68)'` }));
graphics.push(drawText({ value: 'DREAMDOJO · ACTION-CONDITIONED · SYNCHRONIZED ROLLOUT', x: 688, y: 139, size: 17, color: '8298aa', font: FONT_BOLD, enable: `enable='gte(t,5.68)'` }));
tilePositions.forEach(({ x, y }, index) => {
  graphics.push(drawBox({ x, y, w: 279, h: 326, color: '0x0e1c2a@0.98', enable: `enable='gte(t,5.68)'` }));
  graphics.push(drawBox({ x, y, w: 279, h: 326, color: `0x${COLORS[index]}@0.48`, thickness: 2, enable: `enable='gte(t,5.68)'` }));
});

filters.push(`[stage]${graphics.join(',')}[graphics]`);

manifest.universes.slice(0, 8).forEach((_, index) => {
  filters.push(
    `[${index + 1}:v]scale=267:200:force_original_aspect_ratio=increase,crop=267:200,setsar=1,tpad=stop_mode=clone:stop_duration=3.0,setpts=PTS-STARTPTS+${VIDEO_START}/TB,format=yuva420p,fade=t=in:st=${VIDEO_START}:d=0.32:alpha=1[v${index}]`,
  );
});

let previous = 'graphics';
tilePositions.forEach(({ x, y }, index) => {
  const next = `video_wall_${index}`;
  filters.push(`[${previous}][v${index}]overlay=x=${x + 6}:y=${y + 48}:eof_action=pass:enable='gte(t,${VIDEO_START})'[${next}]`);
  previous = next;
});

const captions = [];
tilePositions.forEach(({ x, y }, index) => {
  const universe = manifest.universes[index];
  const color = COLORS[index];
  captions.push(drawBox({ x: x + 15, y: y + 18, w: 9, h: 9, color: `0x${color}@1`, enable: `enable='gte(t,5.68)'` }));
  captions.push(drawText({ value: `UNIVERSE ${String.fromCharCode(65 + index)}`, x: x + 34, y: y + 13, size: 16, color: 'eef6f3', font: FONT_BOLD, enable: `enable='gte(t,5.68)'` }));
  captions.push(drawText({ value: `Δ ACTION ${universe.action_rms_from_universe_0.toFixed(3)}`, x: x + 151, y: y + 17, size: 11, color, font: FONT_BOLD, enable: `enable='gte(t,5.68)'` }));
  universe.action_trace.slice(0, 48).forEach((value, step) => {
    const barHeight = 3 + Math.round((value / 100) * 24);
    captions.push(drawBox({
      x: x + 13 + step * 5.25,
      y: y + 314 - barHeight,
      w: 3,
      h: barHeight,
      color: `0x${color}@0.86`,
      enable: `enable='gte(t,5.68)'`,
    }));
  });
});
captions.push(drawText({ value: '48 ACTION STEPS  ·  30D POLICY OUTPUT  ·  4 × 12 DREAMDOJO WINDOWS', x: 686, y: 958, size: 17, color: '71889b', font: FONT_BOLD, enable: between(VIDEO_START, VIDEO_END) }));
captions.push(drawBox({ x: 650, y: 0, w: 1270, h: 1080, color: '0x03070c@0.72', enable: between(VIDEO_END, DURATION) }));
captions.push(drawText({ value: 'SAME OBSERVATION.', x: 760, y: 388, size: 60, color: 'f2f8f6', font: FONT_BOLD, enable: between(12.72, DURATION) }));
captions.push(drawText({ value: '8 ACTIONS. 8 FUTURES.', x: 760, y: 462, size: 60, color: 'a3ff52', font: FONT_BOLD, enable: between(12.86, DURATION) }));
captions.push(drawText({ value: 'ACTION-CONDITIONED WORLD MODEL', x: 764, y: 565, size: 20, color: '8298aa', font: FONT_BOLD, enable: between(13.05, DURATION) }));
captions.push(drawBox({ x: 760, y: 620, w: 710, h: 4, color: '0xa3ff52@0.86', enable: between(13.12, DURATION) }));

filters.push(`[${previous}]${captions.join(',')},fade=t=out:st=14.82:d=0.38[out]`);

const tempDir = mkdtempSync(join(tmpdir(), 'dreamdojo-showcase-'));
const filterPath = join(tempDir, 'showcase.filter');
writeFileSync(filterPath, `${filters.join(';\n')}\n`, 'utf8');

const args = [
  '-y',
  '-loop', '1',
  '-framerate', String(FPS),
  '-t', String(DURATION),
  '-i', join(artifactDir, 'observation.png'),
];
for (let index = 0; index < 8; index += 1) {
  args.push('-i', join(artifactDir, `universe_${String(index).padStart(2, '0')}.mp4`));
}
args.push(
  '-filter_complex_script', filterPath,
  '-map', '[out]',
  '-an',
  '-c:v', 'libx264',
  '-preset', 'slow',
  '-crf', '21',
  '-pix_fmt', 'yuv420p',
  '-movflags', '+faststart',
  '-t', String(DURATION),
  outputPath,
);

console.log(`Rendering ${outputPath}`);
const result = spawnSync('ffmpeg', args, { cwd: repoRoot, stdio: 'inherit' });
rmSync(tempDir, { recursive: true, force: true });
if (result.status !== 0) process.exit(result.status ?? 1);
console.log('Parallel-universe director cut complete.');
