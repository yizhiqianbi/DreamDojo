'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  BrainCircuit,
  Clapperboard,
  ChevronRight,
  Cpu,
  Download,
  GitBranch,
  Maximize2,
  MousePointer2,
  Play,
  Radio,
  RotateCcw,
  Video,
  ZoomIn,
  ZoomOut,
  Zap,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

type ManifestUniverse = {
  id: number;
  seed: number;
  action_rms_from_universe_0: number;
  pi05_infer_ms: number;
  video: string | null;
  action_trace?: number[];
};

type Manifest = {
  action_horizon: number;
  dreamdojo_chunk_size: number;
  prompt: string;
  source_frame: number;
  universes: ManifestUniverse[];
  pairwise_video_difference?: Record<string, { pixel_psnr_db: number; last_frame_mae: number }>;
};

type Universe = ManifestUniverse & {
  name: string;
  color: string;
  bars: number[];
  videoUrl: string;
  pairwise: string;
};

type Viewport = { x: number; y: number; scale: number };
type Point = { x: number; y: number };

const COLORS = ['#a3ff52', '#52d9ff', '#ffbf5b', '#c68cff', '#ff6f91', '#5ff0c8', '#f7e75b', '#719cff'];
const WORLD = { width: 2020, height: 830 };
const NODE = { width: 254, height: 178 };

function makeTrace(id: number) {
  return Array.from({ length: 48 }, (_, index) => {
    const signal = Math.sin((index + 1) * (0.41 + id * 0.027)) * 0.34;
    const carrier = Math.cos((index + 5) * (0.17 + id * 0.013)) * 0.22;
    return Math.round(24 + Math.abs(signal + carrier) * 112);
  });
}

const fallbackManifest: Manifest = {
  action_horizon: 48,
  dreamdojo_chunk_size: 12,
  prompt: 'arrange the orange juice and green tea neatly',
  source_frame: 400,
  universes: Array.from({ length: 8 }, (_, id) => ({
    id,
    seed: 20260830 + id,
    action_rms_from_universe_0: [0, 0.0991, 0.1094, 0.0778, 0.1049, 0.0881, 0.0934, 0.1047][id],
    pi05_infer_ms: [16478.9, 67.6, 82.3, 81.8, 72.8, 83.2, 84.7, 84.1][id],
    video: `universe_${String(id).padStart(2, '0')}.mp4`,
  })),
};

function nodePosition(id: number): Point {
  return {
    x: 720 + (id % 4) * 316,
    y: 96 + Math.floor(id / 4) * 374,
  };
}

function roundedRect(context: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number) {
  context.beginPath();
  context.roundRect(x, y, width, height, radius);
}

function drawCanvas(
  canvas: HTMLCanvasElement,
  viewport: Viewport,
  universes: Universe[],
  selectedId: number,
) {
  const context = canvas.getContext('2d');
  if (!context) return;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const gradient = context.createRadialGradient(width * 0.56, height * 0.48, 20, width * 0.56, height * 0.48, width * 0.72);
  gradient.addColorStop(0, 'rgba(163,255,82,0.055)');
  gradient.addColorStop(0.55, 'rgba(24,37,53,0.12)');
  gradient.addColorStop(1, 'rgba(7,12,21,0.86)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);

  context.save();
  context.translate(viewport.x, viewport.y);
  context.scale(viewport.scale, viewport.scale);

  context.lineWidth = 1 / viewport.scale;
  context.strokeStyle = 'rgba(123,150,174,0.09)';
  for (let x = 0; x <= WORLD.width; x += 40) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, WORLD.height);
    context.stroke();
  }
  for (let y = 0; y <= WORLD.height; y += 40) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(WORLD.width, y);
    context.stroke();
  }

  const source = { x: 56, y: 305, width: 248, height: 190 };
  const policy = { x: 420, y: 344, width: 174, height: 112 };
  const policyOut = { x: policy.x + policy.width, y: policy.y + policy.height / 2 };

  universes.forEach((universe) => {
    const position = nodePosition(universe.id);
    const target = { x: position.x, y: position.y + NODE.height / 2 };
    const active = universe.id === selectedId;
    const line = context.createLinearGradient(policyOut.x, policyOut.y, target.x, target.y);
    line.addColorStop(0, `${universe.color}38`);
    line.addColorStop(1, active ? universe.color : `${universe.color}72`);
    context.beginPath();
    context.moveTo(policyOut.x, policyOut.y);
    context.bezierCurveTo(policyOut.x + 120, policyOut.y, target.x - 155, target.y, target.x, target.y);
    context.strokeStyle = line;
    context.lineWidth = active ? 4 / viewport.scale : 1.65 / viewport.scale;
    context.stroke();
    context.beginPath();
    context.arc(target.x, target.y, active ? 5.5 : 3.5, 0, Math.PI * 2);
    context.fillStyle = universe.color;
    context.fill();
  });

  roundedRect(context, source.x, source.y, source.width, source.height, 18);
  context.fillStyle = 'rgba(15,24,36,0.98)';
  context.fill();
  context.strokeStyle = 'rgba(118,151,177,0.34)';
  context.lineWidth = 1.2 / viewport.scale;
  context.stroke();
  context.fillStyle = 'rgba(163,255,82,0.12)';
  context.fillRect(source.x + 14, source.y + 15, 52, 5);
  context.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillStyle = '#8ea5b7';
  context.fillText('OBSERVATION / T₀', source.x + 15, source.y + 43);
  context.font = '600 19px Inter, system-ui, sans-serif';
  context.fillStyle = '#eff7f5';
  context.fillText('Jokeru recording', source.x + 15, source.y + 72);
  context.fillStyle = 'rgba(4,8,13,0.84)';
  roundedRect(context, source.x + 14, source.y + 91, source.width - 28, 55, 9);
  context.fill();
  context.font = '500 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillStyle = '#8299aa';
  context.fillText('FRAME 400  ·  7.25 HZ', source.x + 26, source.y + 116);
  context.fillStyle = '#a3ff52';
  context.fillText('RECORDED · NOT GENERATED', source.x + 26, source.y + 135);
  context.font = '500 10px Inter, system-ui, sans-serif';
  context.fillStyle = '#8299aa';
  context.fillText('RGB observation → policy context', source.x + 15, source.y + 172);

  context.beginPath();
  context.moveTo(source.x + source.width, source.y + source.height / 2);
  context.lineTo(policy.x, policy.y + policy.height / 2);
  context.strokeStyle = 'rgba(163,255,82,0.38)';
  context.lineWidth = 2 / viewport.scale;
  context.stroke();

  context.shadowColor = 'rgba(163,255,82,0.26)';
  context.shadowBlur = 34;
  roundedRect(context, policy.x, policy.y, policy.width, policy.height, 18);
  context.fillStyle = 'rgba(18,31,38,0.99)';
  context.fill();
  context.shadowBlur = 0;
  context.strokeStyle = 'rgba(163,255,82,0.62)';
  context.lineWidth = 1.3 / viewport.scale;
  context.stroke();
  context.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillStyle = '#a3ff52';
  context.fillText('STOCHASTIC POLICY', policy.x + 17, policy.y + 29);
  context.font = '700 25px Inter, system-ui, sans-serif';
  context.fillStyle = '#f1f8f5';
  context.fillText('π0.5', policy.x + 17, policy.y + 62);
  context.font = '500 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillStyle = '#8da2af';
  context.fillText('8 NOISE SEEDS · 48×30D', policy.x + 17, policy.y + 88);

  universes.forEach((universe) => {
    const position = nodePosition(universe.id);
    const active = universe.id === selectedId;
    context.shadowColor = active ? `${universe.color}65` : 'transparent';
    context.shadowBlur = active ? 28 : 0;
    roundedRect(context, position.x, position.y, NODE.width, NODE.height, 16);
    context.fillStyle = active ? 'rgba(25,38,48,0.99)' : 'rgba(14,24,36,0.97)';
    context.fill();
    context.shadowBlur = 0;
    context.strokeStyle = active ? universe.color : 'rgba(106,137,161,0.34)';
    context.lineWidth = (active ? 2.1 : 1.05) / viewport.scale;
    context.stroke();

    context.beginPath();
    context.arc(position.x + 18, position.y + 22, 5, 0, Math.PI * 2);
    context.fillStyle = universe.color;
    context.fill();
    context.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.fillStyle = '#8299aa';
    context.fillText(`SAMPLE ${String(universe.id + 1).padStart(2, '0')}`, position.x + 31, position.y + 26);
    context.font = '700 17px Inter, system-ui, sans-serif';
    context.fillStyle = '#eff7f5';
    context.fillText(universe.name, position.x + 15, position.y + 55);
    context.font = '500 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.fillStyle = '#7e94a5';
    context.fillText(`SEED ${universe.seed}  ·  Δ ${universe.action_rms_from_universe_0.toFixed(3)}`, position.x + 15, position.y + 76);

    const traceX = position.x + 15;
    const traceY = position.y + 94;
    const traceWidth = NODE.width - 30;
    const gap = 1.25;
    const barWidth = (traceWidth - gap * 47) / 48;
    universe.bars.forEach((value, index) => {
      const barHeight = 5 + (value / 100) * 34;
      context.fillStyle = active ? universe.color : `${universe.color}9c`;
      context.fillRect(traceX + index * (barWidth + gap), traceY + 38 - barHeight, Math.max(1.2, barWidth), barHeight);
      if ((index + 1) % 12 === 0 && index < 47) {
        context.fillStyle = 'rgba(255,255,255,0.25)';
        context.fillRect(traceX + index * (barWidth + gap) + barWidth + 1.5, traceY - 2, 0.7, 43);
      }
    });
    context.font = '500 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.fillStyle = '#708697';
    context.fillText('4 × 12 ACTION-CONDITIONED ROLLOUT', position.x + 15, position.y + 155);
    context.fillStyle = active ? universe.color : '#708697';
    context.fillText(active ? 'SELECTED · OPEN INSPECTOR' : 'CLICK TO INSPECT', position.x + 15, position.y + 170);
  });

  context.restore();
}

export function ParallelWorldCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ start: Point; origin: Point; moved: boolean } | null>(null);
  const [manifest, setManifest] = useState<Manifest>(fallbackManifest);
  const [selectedId, setSelectedId] = useState(0);
  const [playbackKey, setPlaybackKey] = useState(0);
  const [viewport, setViewport] = useState<Viewport>({ x: 28, y: 16, scale: 0.72 });
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    let mounted = true;
    fetch('/data/pi05-parallel-x8-manifest.json')
      .then((response) => {
        if (!response.ok) throw new Error(`manifest ${response.status}`);
        return response.json() as Promise<Manifest>;
      })
      .then((data) => mounted && setManifest(data))
      .catch(() => undefined);
    return () => {
      mounted = false;
    };
  }, []);

  const universes = useMemo<Universe[]>(() => manifest.universes.slice(0, 8).map((universe) => {
    const pairwise = manifest.pairwise_video_difference?.[`u0_u${universe.id}`]?.pixel_psnr_db;
    return {
      ...universe,
      name: `Universe ${String.fromCharCode(65 + universe.id)}`,
      color: COLORS[universe.id % COLORS.length],
      bars: universe.action_trace?.slice(0, 48) ?? makeTrace(universe.id),
      videoUrl: `/videos/parallel8/universe-${String(universe.id).padStart(2, '0')}.mp4`,
      pairwise: universe.id === 0 ? 'reference branch' : pairwise ? `${pairwise.toFixed(2)} dB vs A` : 'parallel branch',
    };
  }), [manifest]);
  const selected = universes.find((universe) => universe.id === selectedId) ?? universes[0];

  const fitCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const scale = Math.min((canvas.clientWidth - 44) / WORLD.width, (canvas.clientHeight - 44) / WORLD.height);
    setViewport({
      scale: Math.max(0.38, Math.min(1.15, scale)),
      x: (canvas.clientWidth - WORLD.width * scale) / 2,
      y: (canvas.clientHeight - WORLD.height * scale) / 2,
    });
  }, []);

  useEffect(() => {
    fitCanvas();
    const observer = new ResizeObserver(() => fitCanvas());
    if (shellRef.current) observer.observe(shellRef.current);
    return () => observer.disconnect();
  }, [fitCanvas]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawCanvas(canvas, viewport, universes, selectedId);
  }, [selectedId, universes, viewport]);

  const zoom = (factor: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const anchor = { x: canvas.clientWidth / 2, y: canvas.clientHeight / 2 };
    setViewport((current) => {
      const scale = Math.min(1.65, Math.max(0.32, current.scale * factor));
      const worldX = (anchor.x - current.x) / current.scale;
      const worldY = (anchor.y - current.y) / current.scale;
      return { scale, x: anchor.x - worldX * scale, y: anchor.y - worldY * scale };
    });
  };

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { start: { x: event.clientX, y: event.clientY }, origin: { x: viewport.x, y: viewport.y }, moved: false };
    setDragging(true);
  };
  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    const dx = event.clientX - dragRef.current.start.x;
    const dy = event.clientY - dragRef.current.start.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) dragRef.current.moved = true;
    setViewport((current) => ({ ...current, x: dragRef.current!.origin.x + dx, y: dragRef.current!.origin.y + dy }));
  };
  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (drag && !drag.moved) {
      const bounds = event.currentTarget.getBoundingClientRect();
      const worldX = (event.clientX - bounds.left - viewport.x) / viewport.scale;
      const worldY = (event.clientY - bounds.top - viewport.y) / viewport.scale;
      const hit = universes.find((universe) => {
        const position = nodePosition(universe.id);
        return worldX >= position.x && worldX <= position.x + NODE.width && worldY >= position.y && worldY <= position.y + NODE.height;
      });
      if (hit) setSelectedId(hit.id);
    }
    dragRef.current = null;
    setDragging(false);
    event.currentTarget.releasePointerCapture(event.pointerId);
  };
  const onWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const bounds = event.currentTarget.getBoundingClientRect();
    const anchor = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
    setViewport((current) => {
      const scale = Math.min(1.65, Math.max(0.32, current.scale * Math.exp(-event.deltaY * 0.0014)));
      const worldX = (anchor.x - current.x) / current.scale;
      const worldY = (anchor.y - current.y) / current.scale;
      return { scale, x: anchor.x - worldX * scale, y: anchor.y - worldY * scale };
    });
  };

  return (
    <div className="mx-auto max-w-[1760px] space-y-5 px-4 py-5 sm:px-7">
      <section className="panel overflow-hidden">
        <div className="flex flex-col gap-4 border-b border-border/70 px-5 py-5 xl:flex-row xl:items-end xl:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <Badge className="border border-primary/25 bg-primary/10 text-primary"><Radio className="size-3" /> 8 live artifacts</Badge>
              <Badge variant="outline" className="border-chart-2/30 bg-chart-2/8 text-chart-2">recorded-observation mock</Badge>
              <Badge variant="outline" className="border-border bg-secondary/40 text-muted-foreground">π0.5 base · 48-step one-shot</Badge>
            </div>
            <h1 className="text-2xl font-semibold tracking-[-0.025em] sm:text-3xl">One observation. Eight possible futures.</h1>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
              Eight π0.5 flow-matching samples branch from one real Jokeru observation. Each 48-step plan drives four chained DreamDojo windows to form a 49-frame parallel future.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="outline" onClick={() => setPlaybackKey((value) => value + 1)} className="border-border bg-secondary/35"><RotateCcw className="size-3.5" /> Replay</Button>
            <Button className="bg-primary text-primary-foreground hover:bg-primary/90" onClick={() => setSelectedId((selectedId + 1) % universes.length)}><GitBranch className="size-3.5" /> Next universe</Button>
          </div>
        </div>

        <section className="director-cut-shell" aria-labelledby="director-cut-title">
          <div className="director-cut-heading">
            <div className="flex items-start gap-3">
              <div className="stat-icon"><Clapperboard /></div>
              <div>
                <p className="eyebrow">5-layer GT path · 46.5 seconds</p>
                <h2 id="director-cut-title" className="mt-1 text-base font-semibold sm:text-lg">8 candidates → select GT → continue</h2>
                <p className="mt-1 max-w-3xl text-[11px] leading-relaxed text-muted-foreground sm:text-xs">
                  Five receding-horizon stages. Each stage rolls out seven model futures plus one hidden recorded future, then follows the revealed GT endpoint.
                </p>
              </div>
            </div>
            <Button
              size="sm"
              variant="outline"
              render={<a href="/videos/parallel8/parallel-universe-directors-cut.mp4" download />}
              className="border-border bg-secondary/35"
            >
              <Download className="size-3.5" /> Download MP4
            </Button>
          </div>
          <div className="director-video-frame">
            <video
              src="/videos/parallel8/parallel-universe-directors-cut.mp4"
              controls
              autoPlay
              muted
              loop
              playsInline
              preload="metadata"
              className="aspect-video w-full bg-black object-contain"
              aria-label="DreamDojo five-layer ground-truth path rollout"
            />
            <div className="director-video-chip"><span /> 5 layers · 240 action steps</div>
          </div>
        </section>

        <div className="grid grid-cols-2 border-b border-border/70 lg:grid-cols-5">
          {[
            ['Observation', 'recorded RGB', Video],
            ['Policy', 'π0.5 · 2B', Cpu],
            ['Parallel plan', '8 × 48 × 30D', GitBranch],
            ['Conditioning', '8 × 48 × 384D', Activity],
            ['World model', '8 × 4 × 12', BrainCircuit],
          ].map(([label, value, Icon], index) => (
            <div key={label as string} className={cn('pipeline-step', index === 4 && 'col-span-2 lg:col-span-1')}>
              <span className="pipeline-index">{String(index + 1).padStart(2, '0')}</span>
              <Icon className="size-4 text-primary" />
              <div><p className="eyebrow">{label as string}</p><p className="mt-1 text-xs font-medium">{value as string}</p></div>
              {index < 4 && <ChevronRight className="ml-auto hidden size-3.5 text-muted-foreground lg:block" />}
            </div>
          ))}
        </div>

        <div className="canvas-workspace-grid">
          <div ref={shellRef} className="parallel-canvas-shell">
            <canvas
              ref={canvasRef}
              className={cn('parallel-canvas', dragging && 'is-dragging')}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={() => { dragRef.current = null; setDragging(false); }}
              onWheel={onWheel}
              aria-label="Interactive canvas showing one observation branching into eight π0.5 action plans and DreamDojo futures"
            />
            <div className="canvas-toolbar">
              <Button size="icon" variant="outline" onClick={() => zoom(1.18)} aria-label="Zoom in"><ZoomIn /></Button>
              <Button size="icon" variant="outline" onClick={() => zoom(1 / 1.18)} aria-label="Zoom out"><ZoomOut /></Button>
              <Button size="icon" variant="outline" onClick={fitCanvas} aria-label="Fit canvas"><Maximize2 /></Button>
            </div>
            <div className="canvas-hint"><MousePointer2 className="size-3" /> drag to pan · scroll to zoom · click a universe</div>
            <div className="canvas-scale">{Math.round(viewport.scale * 100)}%</div>
          </div>

          <aside className="universe-inspector">
            <div className="flex items-start justify-between gap-3 p-4">
              <div className="flex items-center gap-3"><span className="branch-dot" style={{ background: selected.color, boxShadow: `0 0 14px ${selected.color}` }} /><div><p className="eyebrow">Focused branch</p><h2 className="mt-1 text-base font-semibold">{selected.name}</h2></div></div>
              <Badge variant="outline" style={{ borderColor: `${selected.color}66`, color: selected.color }}>seed {selected.seed}</Badge>
            </div>
            <div className="relative overflow-hidden bg-black">
              <video key={`${selected.id}-${playbackKey}`} src={selected.videoUrl} controls autoPlay muted loop playsInline className="aspect-[4/3] w-full object-cover" />
              <div className="video-label"><Play className="size-3" /> 49-frame generated future</div>
            </div>
            <div className="grid grid-cols-2 gap-2 p-4">
              <div className="mini-readout"><span>action delta</span><strong>{selected.action_rms_from_universe_0.toFixed(4)}</strong></div>
              <div className="mini-readout"><span>video delta</span><strong>{selected.pairwise}</strong></div>
              <div className="mini-readout"><span>VLA latency</span><strong>{selected.id === 0 ? 'JIT warmup' : `${selected.pi05_infer_ms.toFixed(1)} ms`}</strong></div>
              <div className="mini-readout"><span>rollout</span><strong>4 × 12 → 49f</strong></div>
            </div>
            <div className="border-t border-border/60 p-4">
              <p className="eyebrow">Recorded observation</p>
              <video key={`obs-${playbackKey}`} src="/videos/parallel8/observation-replay.mp4" autoPlay muted loop playsInline controls className="mt-3 aspect-[16/5] w-full rounded-lg border border-border/70 bg-black object-cover" />
              <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground">Frame {manifest.source_frame} is replayed from Jokeru; only the eight futures are generated.</p>
            </div>
          </aside>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[1.05fr_0.95fr]">
        <div className="panel p-5">
          <div className="flex items-start justify-between gap-3"><div><p className="eyebrow">Parallel rollout</p><h2 className="mt-1 text-base font-semibold">8 sampled action universes</h2></div><Badge className="border border-primary/25 bg-primary/10 text-primary">392 generated frames</Badge></div>
          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="diagnostic"><span>π0.5 samples</span><strong>8</strong><small>independent Gaussian seeds</small></div>
            <div className="diagnostic"><span>VLA output</span><strong>48 × 30D</strong><small>per parallel branch</small></div>
            <div className="diagnostic"><span>ACWM input</span><strong>48 × 384D</strong><small>per parallel branch</small></div>
            <div className="diagnostic"><span>DreamDojo calls</span><strong>32</strong><small>8 branches × 4 windows</small></div>
          </div>
        </div>
        <div className="panel p-5">
          <div className="flex items-start gap-3"><div className="stat-icon"><Zap /></div><div><p className="eyebrow">Self-Forcing acceleration</p><h2 className="mt-1 text-base font-semibold">Jokeru distillation pipeline wired</h2></div></div>
          <div className="mt-4 grid grid-cols-3 gap-2">
            <div className="self-force-stage"><span>01</span><strong>Teacher cache</strong><small>35-step trajectories</small></div>
            <div className="self-force-stage"><span>02</span><strong>Causal warmup</strong><small>4 denoising states</small></div>
            <div className="self-force-stage"><span>03</span><strong>Self-Forcing</strong><small>DMD autoregression</small></div>
          </div>
          <p className="mt-4 text-[11px] leading-relaxed text-muted-foreground">The official student path is integrated for the Jokeru 384D action condition. The current videos remain teacher outputs until a distilled student checkpoint is trained.</p>
        </div>
      </section>

      <footer className="flex flex-col gap-2 px-1 pb-3 text-[10px] text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
        <span>Recorded observation → π0.5 flow matching → DreamDojo action-conditioned futures</span>
        <span className="font-mono">8 SEEDS · 384 ACTIONS · 392 FRAMES · 7.25 FPS</span>
      </footer>
    </div>
  );
}
