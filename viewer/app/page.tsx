'use client';

import { useMemo, useState } from 'react';
import {
  Activity,
  BrainCircuit,
  Check,
  ChevronRight,
  FlaskConical,
  Gauge,
  Search,
  Sparkles,
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

type Metrics = {
  psnr: number;
  ssim: number;
  lpips: number;
};

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

const evaluations: Evaluation[] = [
  {
    id: 0,
    name: 'Arrange patch',
    shortName: 'Arrange patch',
    family: 'orange juice + green tea',
    video: '/videos/normal-0000.mp4',
    metrics: { psnr: 28.286, ssim: 0.958, lpips: 0.062 },
  },
  {
    id: 1,
    name: 'Arrange patch · annotated',
    shortName: 'Patch annotated',
    family: 'orange juice + green tea',
    video: '/videos/normal-0001.mp4',
    metrics: { psnr: 28.286, ssim: 0.958, lpips: 0.062 },
  },
  {
    id: 2,
    name: 'Grip bottle cap',
    shortName: 'Grip bottle cap',
    family: 'bimanual manipulation',
    video: '/videos/normal-0002.mp4',
    metrics: { psnr: 27.279, ssim: 0.933, lpips: 0.137 },
  },
  {
    id: 3,
    name: 'Arrange set 2',
    shortName: 'Arrange set 2',
    family: 'orange juice + green tea',
    video: '/videos/normal-0003.mp4',
    metrics: { psnr: 25.061, ssim: 0.918, lpips: 0.092 },
    zeroVideo: '/videos/zero-arrange-2.mp4',
    zeroMetrics: { psnr: 18.663, ssim: 0.832, lpips: 0.159 },
  },
  {
    id: 4,
    name: 'Arrange set 2 · annotated',
    shortName: 'Set 2 annotated',
    family: 'orange juice + green tea',
    video: '/videos/normal-0004.mp4',
    metrics: { psnr: 25.061, ssim: 0.918, lpips: 0.092 },
  },
  {
    id: 5,
    name: 'Arrange set 3 · annotated',
    shortName: 'Set 3 annotated',
    family: 'orange juice + green tea',
    video: '/videos/normal-0005.mp4',
    metrics: { psnr: 27.859, ssim: 0.951, lpips: 0.075 },
  },
  {
    id: 6,
    name: 'Continuous shelf organizing',
    shortName: 'Continuous shelf',
    family: 'long-horizon organization',
    video: '/videos/normal-0006.mp4',
    metrics: { psnr: 24.127, ssim: 0.906, lpips: 0.102 },
    zeroVideo: '/videos/zero-continuous-shelf.mp4',
    zeroMetrics: { psnr: 21.053, ssim: 0.871, lpips: 0.124 },
  },
  {
    id: 7,
    name: 'Arrange base',
    shortName: 'Arrange base',
    family: 'orange juice + green tea',
    video: '/videos/normal-0007.mp4',
    metrics: { psnr: 27.608, ssim: 0.95, lpips: 0.074 },
  },
  {
    id: 8,
    name: 'Arrange base · annotated',
    shortName: 'Base annotated',
    family: 'orange juice + green tea',
    video: '/videos/normal-0008.mp4',
    metrics: { psnr: 27.608, ssim: 0.95, lpips: 0.074 },
  },
  {
    id: 9,
    name: 'Arrange set 3',
    shortName: 'Arrange set 3',
    family: 'orange juice + green tea',
    video: '/videos/normal-0009.mp4',
    metrics: { psnr: 27.859, ssim: 0.951, lpips: 0.075 },
  },
  {
    id: 10,
    name: 'Pick up black bottle',
    shortName: 'Pick black bottle',
    family: 'object pickup',
    video: '/videos/normal-0010.mp4',
    metrics: { psnr: 34.978, ssim: 0.976, lpips: 0.117 },
  },
  {
    id: 11,
    name: 'Take wrong item · right arm',
    shortName: 'Wrong item · right',
    family: 'single-arm correction',
    video: '/videos/normal-0011.mp4',
    metrics: { psnr: 24.76, ssim: 0.947, lpips: 0.077 },
  },
  {
    id: 12,
    name: 'Pick purple box → middle',
    shortName: 'Purple box → middle',
    family: 'pick and place',
    video: '/videos/normal-0012.mp4',
    metrics: { psnr: 26.052, ssim: 0.944, lpips: 0.073 },
  },
];

const aggregate = { psnr: 27.294, ssim: 0.943, lpips: 0.086 };

function Metric({ label, value, kind }: { label: string; value: number; kind: keyof Metrics }) {
  const progress =
    kind === 'psnr'
      ? Math.min((value / 40) * 100, 100)
      : kind === 'ssim'
        ? value * 100
        : Math.max(0, (1 - value / 0.2) * 100);

  return (
    <div className="metric-cell">
      <div className="mb-2 flex items-end justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
          {label}
        </span>
        <span className="font-mono text-xl font-semibold tracking-tight text-foreground">
          {kind === 'psnr' ? value.toFixed(2) : value.toFixed(3)}
        </span>
      </div>
      <Progress value={progress} className="[&_[data-slot=progress-indicator]]:bg-chart-1" />
      <p className="mt-2 text-[10px] text-muted-foreground">
        {kind === 'lpips' ? 'lower is better' : 'higher is better'}
      </p>
    </div>
  );
}

export default function Home() {
  const [selectedId, setSelectedId] = useState(6);
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'action' | 'zero'>('action');

  const selected = evaluations.find((item) => item.id === selectedId) ?? evaluations[0];
  const visibleEvaluations = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return evaluations;
    return evaluations.filter((item) =>
      `${item.name} ${item.family}`.toLowerCase().includes(normalized),
    );
  }, [query]);

  const activeMetrics = mode === 'zero' && selected.zeroMetrics ? selected.zeroMetrics : selected.metrics;
  const activeVideo = mode === 'zero' && selected.zeroVideo ? selected.zeroVideo : selected.video;
  const actionDelta = selected.zeroMetrics
    ? selected.metrics.psnr - selected.zeroMetrics.psnr
    : null;

  const chooseEvaluation = (id: number) => {
    setSelectedId(id);
    setMode('action');
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="lab-grid pointer-events-none fixed inset-0 opacity-40" />
      <header className="sticky top-0 z-30 border-b border-border/80 bg-background/88 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-[1680px] items-center justify-between px-4 sm:px-7">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid size-9 shrink-0 place-items-center rounded-xl border border-primary/30 bg-primary/10 text-primary shadow-[0_0_30px_color-mix(in_oklch,var(--primary)_22%,transparent)]">
              <BrainCircuit className="size-5" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-semibold tracking-tight">DreamDojo</span>
                <span className="text-border">/</span>
                <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                  Jokeru eval
                </span>
              </div>
              <p className="truncate text-[11px] text-muted-foreground">2B action-conditioned world model</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="hidden border-border/80 bg-card/50 font-mono text-muted-foreground sm:inline-flex">
              iter_000003000
            </Badge>
            <Badge className="border border-primary/25 bg-primary/10 text-primary">
              <span className="size-1.5 rounded-full bg-primary shadow-[0_0_8px_var(--primary)]" />
              EMA inference
            </Badge>
          </div>
        </div>
      </header>

      <div className="relative mx-auto grid max-w-[1680px] grid-cols-1 gap-5 px-4 py-5 lg:grid-cols-[276px_minmax(0,1fr)] lg:px-7">
        <aside className="panel h-fit p-3 lg:sticky lg:top-[84px]">
          <div className="mb-3 flex items-center justify-between px-2 pt-1">
            <div>
              <p className="eyebrow">Evaluation set</p>
              <p className="mt-1 text-sm font-medium">13 Jokeru tasks</p>
            </div>
            <Badge variant="secondary" className="font-mono">13/13</Badge>
          </div>
          <div className="relative mb-3">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Filter tasks"
              aria-label="Filter evaluation tasks"
              className="h-9 border-border/80 bg-secondary/35 pl-9 text-xs"
            />
          </div>
          <nav aria-label="Evaluation tasks" className="max-h-[calc(100vh-200px)] space-y-1 overflow-y-auto pr-1">
            {visibleEvaluations.map((item) => (
              <Button
                key={item.id}
                variant="ghost"
                onClick={() => chooseEvaluation(item.id)}
                className={cn(
                  'h-auto w-full justify-start gap-3 rounded-xl px-3 py-2.5 text-left',
                  item.id === selected.id
                    ? 'border border-primary/20 bg-primary/10 text-foreground hover:bg-primary/12'
                    : 'border border-transparent text-muted-foreground hover:bg-secondary/50 hover:text-foreground',
                )}
              >
                <span
                  className={cn(
                    'grid size-7 shrink-0 place-items-center rounded-lg border font-mono text-[10px]',
                    item.id === selected.id
                      ? 'border-primary/30 bg-primary/10 text-primary'
                      : 'border-border bg-card text-muted-foreground',
                  )}
                >
                  {String(item.id + 1).padStart(2, '0')}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-xs font-medium">{item.shortName}</span>
                  <span className="mt-0.5 block truncate text-[10px] font-normal text-muted-foreground">
                    {item.family}
                  </span>
                </span>
                <ChevronRight className={cn('size-3.5 opacity-0', item.id === selected.id && 'opacity-100 text-primary')} />
              </Button>
            ))}
          </nav>
        </aside>

        <section className="min-w-0 space-y-5">
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
            <div className="stat-card">
              <div className="stat-icon"><Activity /></div>
              <div><p className="eyebrow">Checkpoint</p><p className="stat-value">3,000</p></div>
            </div>
            <div className="stat-card">
              <div className="stat-icon"><Gauge /></div>
              <div><p className="eyebrow">Mean PSNR</p><p className="stat-value">{aggregate.psnr}</p></div>
            </div>
            <div className="stat-card">
              <div className="stat-icon"><Sparkles /></div>
              <div><p className="eyebrow">Mean SSIM</p><p className="stat-value">{aggregate.ssim}</p></div>
            </div>
            <div className="stat-card">
              <div className="stat-icon"><Zap /></div>
              <div><p className="eyebrow">Mean LPIPS</p><p className="stat-value">{aggregate.lpips}</p></div>
            </div>
          </div>

          <section className="panel overflow-hidden">
            <div className="flex flex-col gap-3 border-b border-border/70 px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-5">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h1 className="truncate text-lg font-semibold tracking-tight sm:text-xl">{selected.name}</h1>
                  {selected.zeroVideo && (
                    <Badge variant="outline" className="border-chart-3/40 bg-chart-3/10 text-chart-3">
                      action ablation ready
                    </Badge>
                  )}
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  First frame + 12 actions → 13-frame video · 480×640 · EMA bf16
                </p>
              </div>
              <div className="flex rounded-xl border border-border bg-secondary/30 p-1">
                <Button
                  size="sm"
                  variant={mode === 'action' ? 'default' : 'ghost'}
                  onClick={() => setMode('action')}
                  className={cn('rounded-lg px-3', mode === 'action' && 'bg-primary text-primary-foreground hover:bg-primary/90')}
                >
                  Real action
                </Button>
                <Button
                  size="sm"
                  variant={mode === 'zero' ? 'default' : 'ghost'}
                  disabled={!selected.zeroVideo}
                  onClick={() => setMode('zero')}
                  className={cn('rounded-lg px-3', mode === 'zero' && 'bg-chart-3 text-background hover:bg-chart-3/90')}
                >
                  Zero action
                </Button>
              </div>
            </div>

            <div className="relative bg-black">
              <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex justify-between bg-gradient-to-b from-black/75 to-transparent px-4 pb-8 pt-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-white/75 sm:px-6">
                <span>Ground truth</span>
                <span>Prediction</span>
              </div>
              <video
                key={`${selected.id}-${mode}`}
                src={activeVideo}
                controls
                autoPlay
                muted
                loop
                playsInline
                preload="metadata"
                aria-label={`${selected.name}, ground truth on the left and model prediction on the right`}
                className="aspect-[8/3] w-full object-contain"
              />
            </div>

            <div className="grid grid-cols-1 border-t border-border/70 md:grid-cols-[1fr_1fr_1fr_260px]">
              <Metric label="PSNR" value={activeMetrics.psnr} kind="psnr" />
              <Metric label="SSIM" value={activeMetrics.ssim} kind="ssim" />
              <Metric label="LPIPS" value={activeMetrics.lpips} kind="lpips" />
              <div className="flex min-h-[116px] flex-col justify-between border-l-0 border-border/70 p-4 md:border-l">
                <div className="flex items-center gap-2 text-xs font-medium">
                  <FlaskConical className="size-4 text-chart-3" />
                  Action sensitivity
                </div>
                {actionDelta !== null ? (
                  <div>
                    <p className="font-mono text-2xl font-semibold text-chart-3">+{actionDelta.toFixed(2)} dB</p>
                    <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">Real actions outperform zero-action conditioning on this sample.</p>
                  </div>
                ) : (
                  <p className="text-[11px] leading-relaxed text-muted-foreground">Select an ablation-ready task to inspect causal action response.</p>
                )}
              </div>
            </div>
          </section>

          <section className="panel overflow-hidden">
            <div className="flex items-center justify-between border-b border-border/70 px-5 py-4">
              <div>
                <p className="eyebrow">All evaluations</p>
                <h2 className="mt-1 text-sm font-semibold">Cross-task scorecard</h2>
              </div>
              <div className="hidden items-center gap-2 text-[11px] text-muted-foreground sm:flex">
                <Check className="size-3.5 text-primary" />
                13 videos validated
              </div>
            </div>
            <Table>
              <TableHeader>
                <TableRow className="border-border/70 hover:bg-transparent">
                  <TableHead className="pl-5 text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Task</TableHead>
                  <TableHead className="text-right text-[10px] uppercase tracking-[0.14em] text-muted-foreground">PSNR</TableHead>
                  <TableHead className="text-right text-[10px] uppercase tracking-[0.14em] text-muted-foreground">SSIM</TableHead>
                  <TableHead className="pr-5 text-right text-[10px] uppercase tracking-[0.14em] text-muted-foreground">LPIPS</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {evaluations.map((item) => (
                  <TableRow key={item.id} className={cn('border-border/55', item.id === selected.id && 'bg-primary/[0.055]')}>
                    <TableCell className="pl-3 sm:pl-5">
                      <Button variant="ghost" onClick={() => chooseEvaluation(item.id)} className="h-auto justify-start px-2 py-1.5 text-left">
                        <span className="font-mono text-[10px] text-muted-foreground">{String(item.id + 1).padStart(2, '0')}</span>
                        <span className="max-w-[230px] truncate text-xs">{item.name}</span>
                      </Button>
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">{item.metrics.psnr.toFixed(2)}</TableCell>
                    <TableCell className="text-right font-mono text-xs">{item.metrics.ssim.toFixed(3)}</TableCell>
                    <TableCell className="pr-5 text-right font-mono text-xs">{item.metrics.lpips.toFixed(3)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </section>

          <footer className="flex flex-col gap-2 px-1 pb-3 text-[10px] text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
            <span>DreamDojo 2B · Jokeru post-training · effective 0.707 epoch</span>
            <span className="font-mono">GT LEFT / PREDICTION RIGHT · 10 FPS</span>
          </footer>
        </section>
      </div>
    </main>
  );
}
