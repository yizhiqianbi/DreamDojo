'use client';

import type { CSSProperties } from 'react';
import { useMemo, useState } from 'react';
import {
  Activity,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleDot,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
  GitBranch,
  Layers3,
  Play,
  Radio,
  RotateCcw,
  Search,
  Sparkles,
  Timer,
  Video,
  Zap,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { cn } from '@/lib/utils';

type Metrics = { psnr: number; ssim: number; lpips: number };

type Evaluation = {
  id: number;
  name: string;
  shortName: string;
  family: string;
  video: string;
  metrics: Metrics;
  zeroVideo?: string;
  zeroMetrics?: Metrics;
};

type Universe = {
  id: number;
  name: string;
  seed: number;
  color: string;
  video: string;
  actionRms: number;
  inferMs: number;
  pairwise: string;
  bars: number[];
};

const universes: Universe[] = [
  {
    id: 0,
    name: 'Universe A',
    seed: 20260830,
    color: '#a3ff52',
    video: '/videos/parallel/universe-00.mp4',
    actionRms: 0,
    inferMs: 15336.5,
    pairwise: 'reference branch',
    bars: [18, 90, 61, 38, 71, 64, 66, 41, 92, 61, 79, 57, 65, 31, 47, 51, 76, 31, 83, 64, 51, 29, 43, 56, 44, 59, 33, 29, 28, 42, 70, 91, 85, 56, 52, 53, 56, 24, 47, 19, 73, 40, 18, 48, 46, 18, 72, 78],
  },
  {
    id: 1,
    name: 'Universe B',
    seed: 20260831,
    color: '#52d9ff',
    video: '/videos/parallel/universe-01.mp4',
    actionRms: 0.0986,
    inferMs: 74.2,
    pairwise: '18.12 dB vs A',
    bars: [18, 69, 87, 52, 40, 50, 40, 31, 34, 46, 40, 23, 36, 37, 54, 19, 87, 18, 36, 64, 34, 52, 55, 56, 33, 54, 41, 67, 45, 88, 26, 18, 18, 78, 34, 68, 73, 91, 85, 47, 66, 43, 92, 43, 83, 56, 75, 85],
  },
  {
    id: 2,
    name: 'Universe C',
    seed: 20260832,
    color: '#ffbf5b',
    video: '/videos/parallel/universe-02.mp4',
    actionRms: 0.1094,
    inferMs: 81.5,
    pairwise: '18.15 dB vs A',
    bars: [18, 85, 92, 32, 47, 57, 56, 52, 46, 49, 55, 22, 34, 43, 89, 92, 66, 36, 21, 37, 29, 25, 27, 46, 49, 26, 30, 18, 34, 30, 22, 22, 31, 31, 42, 18, 18, 48, 70, 46, 45, 37, 37, 41, 42, 36, 52, 56],
  },
];

const evaluations: Evaluation[] = [
  { id: 0, name: 'Arrange patch', shortName: 'Arrange patch', family: 'orange juice + green tea', video: '/videos/normal-0000.mp4', metrics: { psnr: 28.286, ssim: 0.958, lpips: 0.062 } },
  { id: 1, name: 'Arrange patch · annotated', shortName: 'Patch annotated', family: 'orange juice + green tea', video: '/videos/normal-0001.mp4', metrics: { psnr: 28.286, ssim: 0.958, lpips: 0.062 } },
  { id: 2, name: 'Grip bottle cap', shortName: 'Grip bottle cap', family: 'bimanual manipulation', video: '/videos/normal-0002.mp4', metrics: { psnr: 27.279, ssim: 0.933, lpips: 0.137 } },
  { id: 3, name: 'Arrange set 2', shortName: 'Arrange set 2', family: 'orange juice + green tea', video: '/videos/normal-0003.mp4', metrics: { psnr: 25.061, ssim: 0.918, lpips: 0.092 }, zeroVideo: '/videos/zero-arrange-2.mp4', zeroMetrics: { psnr: 18.663, ssim: 0.832, lpips: 0.159 } },
  { id: 4, name: 'Arrange set 2 · annotated', shortName: 'Set 2 annotated', family: 'orange juice + green tea', video: '/videos/normal-0004.mp4', metrics: { psnr: 25.061, ssim: 0.918, lpips: 0.092 } },
  { id: 5, name: 'Arrange set 3 · annotated', shortName: 'Set 3 annotated', family: 'orange juice + green tea', video: '/videos/normal-0005.mp4', metrics: { psnr: 27.859, ssim: 0.951, lpips: 0.075 } },
  { id: 6, name: 'Continuous shelf organizing', shortName: 'Continuous shelf', family: 'long-horizon organization', video: '/videos/normal-0006.mp4', metrics: { psnr: 24.127, ssim: 0.906, lpips: 0.102 }, zeroVideo: '/videos/zero-continuous-shelf.mp4', zeroMetrics: { psnr: 21.053, ssim: 0.871, lpips: 0.124 } },
  { id: 7, name: 'Arrange base', shortName: 'Arrange base', family: 'orange juice + green tea', video: '/videos/normal-0007.mp4', metrics: { psnr: 27.608, ssim: 0.95, lpips: 0.074 } },
  { id: 8, name: 'Arrange base · annotated', shortName: 'Base annotated', family: 'orange juice + green tea', video: '/videos/normal-0008.mp4', metrics: { psnr: 27.608, ssim: 0.95, lpips: 0.074 } },
  { id: 9, name: 'Arrange set 3', shortName: 'Arrange set 3', family: 'orange juice + green tea', video: '/videos/normal-0009.mp4', metrics: { psnr: 27.859, ssim: 0.951, lpips: 0.075 } },
  { id: 10, name: 'Pick up black bottle', shortName: 'Pick black bottle', family: 'object pickup', video: '/videos/normal-0010.mp4', metrics: { psnr: 34.978, ssim: 0.976, lpips: 0.117 } },
  { id: 11, name: 'Take wrong item · right arm', shortName: 'Wrong item · right', family: 'single-arm correction', video: '/videos/normal-0011.mp4', metrics: { psnr: 24.76, ssim: 0.947, lpips: 0.077 } },
  { id: 12, name: 'Pick purple box → middle', shortName: 'Purple box → middle', family: 'pick and place', video: '/videos/normal-0012.mp4', metrics: { psnr: 26.052, ssim: 0.944, lpips: 0.073 } },
];

const aggregate = { psnr: 27.294, ssim: 0.943, lpips: 0.086 };

function Metric({ label, value, kind }: { label: string; value: number; kind: keyof Metrics }) {
  const progress = kind === 'psnr' ? Math.min((value / 40) * 100, 100) : kind === 'ssim' ? value * 100 : Math.max(0, (1 - value / 0.2) * 100);
  return (
    <div className="metric-cell">
      <div className="mb-2 flex items-end justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">{label}</span>
        <span className="font-mono text-xl font-semibold tracking-tight text-foreground">{kind === 'psnr' ? value.toFixed(2) : value.toFixed(3)}</span>
      </div>
      <Progress value={progress} className="[&_[data-slot=progress-indicator]]:bg-chart-1" />
      <p className="mt-2 text-[10px] text-muted-foreground">{kind === 'lpips' ? 'lower is better' : 'higher is better'}</p>
    </div>
  );
}

function ActionTrace({ universe }: { universe: Universe }) {
  return (
    <div className="action-trace" aria-label={`${universe.name} 48-step action magnitude trace`}>
      {universe.bars.map((height, index) => (
        <span key={index} style={{ height: `${height}%`, backgroundColor: universe.color }} />
      ))}
    </div>
  );
}

function ParallelWorkspace() {
  const [selectedUniverse, setSelectedUniverse] = useState(0);
  const [playbackKey, setPlaybackKey] = useState(0);
  const selected = universes[selectedUniverse];

  return (
    <div className="mx-auto max-w-[1760px] space-y-5 px-4 py-5 sm:px-7">
      <section className="panel overflow-hidden">
        <div className="flex flex-col gap-4 border-b border-border/70 px-5 py-5 xl:flex-row xl:items-end xl:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <Badge className="border border-primary/25 bg-primary/10 text-primary"><Radio className="size-3" /> Live artifact</Badge>
              <Badge variant="outline" className="border-chart-2/30 bg-chart-2/8 text-chart-2">recorded-observation mock</Badge>
              <Badge variant="outline" className="border-border bg-secondary/40 text-muted-foreground">π0.5 base · pre-finetune</Badge>
            </div>
            <h1 className="text-2xl font-semibold tracking-[-0.025em] sm:text-3xl">One observation. Three possible futures.</h1>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
              A real Jokeru observation is replayed into π0.5 three times with distinct flow-matching noise seeds. Each 48-step plan is rolled through four chained DreamDojo windows.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="outline" onClick={() => setPlaybackKey((value) => value + 1)} className="border-border bg-secondary/35">
              <RotateCcw className="size-3.5" /> Replay all
            </Button>
            <Button className="bg-primary text-primary-foreground hover:bg-primary/90" onClick={() => setSelectedUniverse((selectedUniverse + 1) % universes.length)}>
              <GitBranch className="size-3.5" /> Focus next branch
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-2 border-b border-border/70 lg:grid-cols-5">
          {[
            ['Observation', 'recorded RGB', Video],
            ['Policy', 'π0.5 · 2B', Cpu],
            ['Action plan', '48 × 30D', Activity],
            ['Conditioning', '48 × 384D', Layers3],
            ['World model', '4 × 12 rollout', BrainCircuit],
          ].map(([label, value, Icon], index) => (
            <div key={label as string} className={cn('pipeline-step', index === 4 && 'col-span-2 lg:col-span-1')}>
              <span className="pipeline-index">{String(index + 1).padStart(2, '0')}</span>
              <Icon className="size-4 text-primary" />
              <div><p className="eyebrow">{label as string}</p><p className="mt-1 text-xs font-medium">{value as string}</p></div>
              {index < 4 && <ChevronRight className="ml-auto hidden size-3.5 text-muted-foreground lg:block" />}
            </div>
          ))}
        </div>

        <div className="parallel-map p-4 sm:p-5">
          <article className="source-card">
            <div className="flex items-start justify-between gap-3 p-4">
              <div><p className="eyebrow">Mock environment · t = 13.79s</p><h2 className="mt-1 text-sm font-semibold">Recorded observation</h2></div>
              <Badge variant="outline" className="border-primary/25 bg-primary/8 font-mono text-primary">frame 400</Badge>
            </div>
            <div className="relative overflow-hidden bg-black">
              <video key={`source-${playbackKey}`} src="/videos/parallel/observation-replay.mp4" autoPlay muted loop playsInline controls className="aspect-[4/3] w-full object-cover" />
              <div className="video-label"><CircleDot className="size-3" /> stolen from dataset · not generated</div>
            </div>
            <div className="space-y-3 p-4">
              <div><p className="eyebrow">Language instruction</p><p className="mt-1 text-xs leading-relaxed">“arrange the orange juice and green tea neatly”</p></div>
              <div className="grid grid-cols-2 gap-2">
                <div className="mini-readout"><span>source</span><strong>Jokeru</strong></div>
                <div className="mini-readout"><span>cadence</span><strong>7.25 Hz</strong></div>
              </div>
            </div>
          </article>

          <div className="branch-engine">
            <div className="policy-node">
              <span className="policy-pulse" />
              <Cpu className="size-5 text-primary" />
              <div><p className="eyebrow">Stochastic policy</p><p className="mt-1 text-sm font-semibold">π0.5</p></div>
              <Badge variant="secondary" className="ml-auto font-mono">×3</Badge>
            </div>
            <svg viewBox="0 0 120 480" preserveAspectRatio="none" aria-hidden="true" className="branch-lines">
              {universes.map((universe, index) => (
                <path key={universe.id} d={`M 0 240 C 56 240, 52 ${80 + index * 160}, 120 ${80 + index * 160}`} fill="none" stroke={universe.color} strokeWidth={selectedUniverse === universe.id ? 3 : 1.4} opacity={selectedUniverse === universe.id ? 1 : 0.42} />
              ))}
            </svg>
            <div className="hidden text-center font-mono text-[9px] uppercase tracking-[0.16em] text-muted-foreground xl:block">different Gaussian noise seeds</div>
          </div>

          <div className="space-y-3">
            {universes.map((universe) => {
              const active = selectedUniverse === universe.id;
              return (
                <article key={universe.id} className={cn('universe-card', active && 'is-active')} style={{ '--branch-color': universe.color } as CSSProperties}>
                  <div className="grid gap-0 md:grid-cols-[minmax(230px,0.92fr)_minmax(260px,1.08fr)]">
                    <div className="flex flex-col justify-between border-b border-border/60 p-4 md:border-b-0 md:border-r">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-center gap-3"><span className="branch-dot" /><div><p className="eyebrow">π0.5 sample {universe.id + 1}</p><h3 className="mt-1 text-sm font-semibold">{universe.name}</h3></div></div>
                        <Button size="sm" variant={active ? 'default' : 'outline'} onClick={() => setSelectedUniverse(universe.id)} className={cn('h-7 rounded-lg px-2.5 text-[10px]', active && 'bg-primary text-primary-foreground')}>{active ? 'Focused' : 'Focus'}</Button>
                      </div>
                      <ActionTrace universe={universe} />
                      <div className="grid grid-cols-3 gap-2">
                        <div className="mini-readout"><span>seed</span><strong>{String(universe.seed).slice(-3)}</strong></div>
                        <div className="mini-readout"><span>Δ action</span><strong>{universe.actionRms.toFixed(3)}</strong></div>
                        <div className="mini-readout"><span>warm infer</span><strong>{universe.id === 0 ? 'JIT' : `${universe.inferMs.toFixed(0)}ms`}</strong></div>
                      </div>
                    </div>
                    <div className="relative overflow-hidden bg-black">
                      <video key={`u-${universe.id}-${playbackKey}`} src={universe.video} autoPlay muted loop playsInline controls className="aspect-[4/3] h-full w-full object-cover" />
                      <div className="video-label"><Play className="size-3" /> 49-frame DreamDojo future · {universe.pairwise}</div>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="panel p-5">
          <div className="flex items-start justify-between gap-4"><div><p className="eyebrow">Focused branch</p><h2 className="mt-1 text-base font-semibold">{selected.name} diagnostics</h2></div><Badge style={{ borderColor: selected.color, color: selected.color }} variant="outline">seed {selected.seed}</Badge></div>
          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="diagnostic"><span>VLA output</span><strong>48 × 30D</strong><small>one-shot absolute controls</small></div>
            <div className="diagnostic"><span>ACWM rollout</span><strong>4 × 12</strong><small>48 × 384D conditions</small></div>
            <div className="diagnostic"><span>Action RMS</span><strong>{selected.actionRms.toFixed(4)}</strong><small>from Universe A</small></div>
            <div className="diagnostic"><span>Video delta</span><strong>{selected.id === 0 ? '—' : selected.pairwise.split(' ')[0]}</strong><small>pixel PSNR vs A</small></div>
          </div>
        </div>
        <div className="panel p-5">
          <div className="flex items-start gap-3"><div className="stat-icon"><FlaskConical /></div><div><p className="eyebrow">Training readiness</p><h2 className="mt-1 text-base font-semibold">Jokeru → π0.5 LoRA verified</h2></div></div>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="mini-readout"><span>episodes</span><strong>592</strong></div>
            <div className="mini-readout"><span>control frames</span><strong>95,359</strong></div>
            <div className="mini-readout"><span>1-step loss</span><strong>0.8082</strong></div>
            <div className="mini-readout"><span>checkpoint</span><strong>saved</strong></div>
          </div>
          <p className="mt-4 text-[11px] leading-relaxed text-muted-foreground">The displayed branches use π0.5 base for a reproducible pre-finetune baseline. A real LoRA gradient step and checkpoint save both completed successfully.</p>
        </div>
      </section>

      <footer className="flex flex-col gap-2 px-1 pb-3 text-[10px] text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
        <span>Recorded observation → π0.5 flow matching → DreamDojo action-conditioned future</span>
        <span className="font-mono">3 SEEDS · 48 ACTIONS · 49 FRAMES · 7.25 FPS</span>
      </footer>
    </div>
  );
}

function EvaluationWorkspace() {
  const [selectedId, setSelectedId] = useState(6);
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'action' | 'zero'>('action');
  const selected = evaluations.find((item) => item.id === selectedId) ?? evaluations[0];
  const visibleEvaluations = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return normalized ? evaluations.filter((item) => `${item.name} ${item.family}`.toLowerCase().includes(normalized)) : evaluations;
  }, [query]);
  const activeMetrics = mode === 'zero' && selected.zeroMetrics ? selected.zeroMetrics : selected.metrics;
  const activeVideo = mode === 'zero' && selected.zeroVideo ? selected.zeroVideo : selected.video;
  const actionDelta = selected.zeroMetrics ? selected.metrics.psnr - selected.zeroMetrics.psnr : null;
  const chooseEvaluation = (id: number) => { setSelectedId(id); setMode('action'); };

  return (
    <div className="relative mx-auto grid max-w-[1680px] grid-cols-1 gap-5 px-4 py-5 lg:grid-cols-[276px_minmax(0,1fr)] lg:px-7">
      <aside className="panel h-fit p-3 lg:sticky lg:top-[84px]">
        <div className="mb-3 flex items-center justify-between px-2 pt-1"><div><p className="eyebrow">Evaluation set</p><p className="mt-1 text-sm font-medium">13 Jokeru tasks</p></div><Badge variant="secondary" className="font-mono">13/13</Badge></div>
        <div className="relative mb-3"><Search className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" /><Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter tasks" aria-label="Filter evaluation tasks" className="h-9 border-border/80 bg-secondary/35 pl-9 text-xs" /></div>
        <nav aria-label="Evaluation tasks" className="max-h-[calc(100vh-200px)] space-y-1 overflow-y-auto pr-1">
          {visibleEvaluations.map((item) => (
            <Button key={item.id} variant="ghost" onClick={() => chooseEvaluation(item.id)} className={cn('h-auto w-full justify-start gap-3 rounded-xl px-3 py-2.5 text-left', item.id === selected.id ? 'border border-primary/20 bg-primary/10 text-foreground' : 'border border-transparent text-muted-foreground hover:bg-secondary/50 hover:text-foreground')}>
              <span className={cn('grid size-7 shrink-0 place-items-center rounded-lg border font-mono text-[10px]', item.id === selected.id ? 'border-primary/30 bg-primary/10 text-primary' : 'border-border bg-card')}>{String(item.id + 1).padStart(2, '0')}</span>
              <span className="min-w-0 flex-1"><span className="block truncate text-xs font-medium">{item.shortName}</span><span className="mt-0.5 block truncate text-[10px] font-normal text-muted-foreground">{item.family}</span></span>
              <ChevronRight className={cn('size-3.5 opacity-0', item.id === selected.id && 'text-primary opacity-100')} />
            </Button>
          ))}
        </nav>
      </aside>

      <section className="min-w-0 space-y-5">
        <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
          <div className="stat-card"><div className="stat-icon"><Activity /></div><div><p className="eyebrow">Checkpoint</p><p className="stat-value">3,000</p></div></div>
          <div className="stat-card"><div className="stat-icon"><Gauge /></div><div><p className="eyebrow">Mean PSNR</p><p className="stat-value">{aggregate.psnr}</p></div></div>
          <div className="stat-card"><div className="stat-icon"><Sparkles /></div><div><p className="eyebrow">Mean SSIM</p><p className="stat-value">{aggregate.ssim}</p></div></div>
          <div className="stat-card"><div className="stat-icon"><Zap /></div><div><p className="eyebrow">Mean LPIPS</p><p className="stat-value">{aggregate.lpips}</p></div></div>
        </div>

        <section className="panel overflow-hidden">
          <div className="flex flex-col gap-3 border-b border-border/70 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
            <div><div className="flex flex-wrap items-center gap-2"><h1 className="text-lg font-semibold tracking-tight sm:text-xl">{selected.name}</h1>{selected.zeroVideo && <Badge variant="outline" className="border-chart-3/40 bg-chart-3/10 text-chart-3">action ablation ready</Badge>}</div><p className="mt-1 text-xs text-muted-foreground">First frame + 12 actions → 13-frame video · 480×640 · EMA bf16</p></div>
            <div className="flex rounded-xl border border-border bg-secondary/30 p-1"><Button size="sm" variant={mode === 'action' ? 'default' : 'ghost'} onClick={() => setMode('action')} className={cn('rounded-lg px-3', mode === 'action' && 'bg-primary text-primary-foreground')}>Real action</Button><Button size="sm" variant={mode === 'zero' ? 'default' : 'ghost'} disabled={!selected.zeroVideo} onClick={() => setMode('zero')} className={cn('rounded-lg px-3', mode === 'zero' && 'bg-chart-3 text-background')}>Zero action</Button></div>
          </div>
          <div className="relative bg-black"><div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex justify-between bg-gradient-to-b from-black/75 to-transparent px-6 pb-8 pt-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-white/75"><span>Ground truth</span><span>Prediction</span></div><video key={`${selected.id}-${mode}`} src={activeVideo} controls autoPlay muted loop playsInline preload="metadata" className="aspect-[8/3] w-full object-contain" /></div>
          <div className="grid grid-cols-1 border-t border-border/70 md:grid-cols-[1fr_1fr_1fr_260px]"><Metric label="PSNR" value={activeMetrics.psnr} kind="psnr" /><Metric label="SSIM" value={activeMetrics.ssim} kind="ssim" /><Metric label="LPIPS" value={activeMetrics.lpips} kind="lpips" /><div className="flex min-h-[116px] flex-col justify-between border-border/70 p-4 md:border-l"><div className="flex items-center gap-2 text-xs font-medium"><FlaskConical className="size-4 text-chart-3" />Action sensitivity</div>{actionDelta !== null ? <div><p className="font-mono text-2xl font-semibold text-chart-3">+{actionDelta.toFixed(2)} dB</p><p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">Real actions outperform zero-action conditioning.</p></div> : <p className="text-[11px] text-muted-foreground">Select an ablation-ready task.</p>}</div></div>
        </section>

        <section className="panel overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/70 px-5 py-4"><div><p className="eyebrow">All evaluations</p><h2 className="mt-1 text-sm font-semibold">Cross-task scorecard</h2></div><div className="hidden items-center gap-2 text-[11px] text-muted-foreground sm:flex"><Check className="size-3.5 text-primary" />13 videos validated</div></div>
          <Table><TableHeader><TableRow className="border-border/70 hover:bg-transparent"><TableHead className="pl-5 text-[10px] uppercase tracking-[0.14em]">Task</TableHead><TableHead className="text-right text-[10px] uppercase tracking-[0.14em]">PSNR</TableHead><TableHead className="text-right text-[10px] uppercase tracking-[0.14em]">SSIM</TableHead><TableHead className="pr-5 text-right text-[10px] uppercase tracking-[0.14em]">LPIPS</TableHead></TableRow></TableHeader><TableBody>{evaluations.map((item) => <TableRow key={item.id} className={cn('border-border/55', item.id === selected.id && 'bg-primary/[0.055]')}><TableCell className="pl-3 sm:pl-5"><Button variant="ghost" onClick={() => chooseEvaluation(item.id)} className="h-auto justify-start px-2 py-1.5 text-left"><span className="font-mono text-[10px] text-muted-foreground">{String(item.id + 1).padStart(2, '0')}</span><span className="max-w-[230px] truncate text-xs">{item.name}</span></Button></TableCell><TableCell className="text-right font-mono text-xs">{item.metrics.psnr.toFixed(2)}</TableCell><TableCell className="text-right font-mono text-xs">{item.metrics.ssim.toFixed(3)}</TableCell><TableCell className="pr-5 text-right font-mono text-xs">{item.metrics.lpips.toFixed(3)}</TableCell></TableRow>)}</TableBody></Table>
        </section>
      </section>
    </div>
  );
}

export default function Home() {
  const [workspace, setWorkspace] = useState<'parallel' | 'evaluation'>('parallel');
  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="lab-grid pointer-events-none fixed inset-0 opacity-40" />
      <header className="sticky top-0 z-30 border-b border-border/80 bg-background/88 backdrop-blur-xl">
        <div className="mx-auto flex min-h-16 max-w-[1760px] flex-wrap items-center justify-between gap-3 px-4 py-2 sm:px-7">
          <div className="flex min-w-0 items-center gap-3"><div className="grid size-9 shrink-0 place-items-center rounded-xl border border-primary/30 bg-primary/10 text-primary shadow-[0_0_30px_color-mix(in_oklch,var(--primary)_22%,transparent)]"><BrainCircuit className="size-5" /></div><div className="min-w-0"><div className="flex items-center gap-2"><span className="truncate text-sm font-semibold tracking-tight">DreamDojo</span><span className="text-border">/</span><span className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Parallel Lab</span></div><p className="truncate text-[11px] text-muted-foreground">π0.5 × action-conditioned world model</p></div></div>
          <div className="flex items-center gap-2 rounded-xl border border-border bg-secondary/25 p-1"><Button size="sm" variant={workspace === 'parallel' ? 'default' : 'ghost'} onClick={() => setWorkspace('parallel')} className={cn('h-8 rounded-lg', workspace === 'parallel' && 'bg-primary text-primary-foreground')}><GitBranch className="size-3.5" />Parallel worlds</Button><Button size="sm" variant={workspace === 'evaluation' ? 'default' : 'ghost'} onClick={() => setWorkspace('evaluation')} className={cn('h-8 rounded-lg', workspace === 'evaluation' && 'bg-primary text-primary-foreground')}><Database className="size-3.5" />Evaluation</Button></div>
          <div className="hidden items-center gap-2 xl:flex"><Badge variant="outline" className="border-border/80 bg-card/50 font-mono text-muted-foreground"><Timer className="size-3" /> warm VLA 74–81ms</Badge><Badge className="border border-primary/25 bg-primary/10 text-primary"><span className="size-1.5 rounded-full bg-primary shadow-[0_0_8px_var(--primary)]" />LoRA smoke passed</Badge></div>
        </div>
      </header>
      {workspace === 'parallel' ? <ParallelWorkspace /> : <EvaluationWorkspace />}
    </main>
  );
}
