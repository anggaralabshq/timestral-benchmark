import React from 'react';
import {AbsoluteFill} from 'remotion';
import {mono, sans} from './theme';

export type Mode = 'light' | 'dark';
export type StillProps = {mode: Mode};

const T = {
  dark: {
    bg: '#0A0E17', panel: '#101624', border: '#232C3F', text: '#E8ECF3', muted: '#8A94A8',
    accent: '#FB923C', accentSoft: 'rgba(251,146,60,0.12)', cool: '#60A5FA',
    glow: 'radial-gradient(1200px 500px at 20% 0%, rgba(251,146,60,0.16), transparent 60%)',
  },
  light: {
    bg: '#FFFFFF', panel: '#F5F7FA', border: '#D8DEE8', text: '#0F172A', muted: '#526077',
    accent: '#C2410C', accentSoft: 'rgba(194,65,12,0.08)', cool: '#1D4ED8',
    glow: 'radial-gradient(1200px 500px at 20% 0%, rgba(194,65,12,0.08), transparent 60%)',
  },
} as const;

const MODELS = ['Timestral 209M · 5M', 'Chronos-2', 'TimesFM 2.5', 'AutoARIMA', 'Prophet', 'ETS', 'Seasonal naive'];

const Wordmark: React.FC<{size: number; mode: Mode}> = ({size, mode}) => (
  <div style={{fontFamily: mono, fontWeight: 700, fontSize: size, letterSpacing: -size * 0.02, lineHeight: 1}}>
    <span style={{color: T[mode].accent}}>›</span>
    <span style={{color: T[mode].text}}>timestral-benchmark</span>
  </div>
);

const Chip: React.FC<{mode: Mode; hi?: boolean; children: React.ReactNode; size?: number}> = ({mode, hi, children, size = 24}) => (
  <div
    style={{
      fontFamily: sans, fontWeight: 600, fontSize: size, whiteSpace: 'nowrap',
      color: hi ? T[mode].accent : T[mode].text,
      background: hi ? T[mode].accentSoft : T[mode].panel,
      border: `2px solid ${hi ? T[mode].accent : T[mode].border}`,
      borderRadius: 999, padding: `${size * 0.35}px ${size * 0.8}px`,
    }}
  >
    {children}
  </div>
);

/* ------------------------------------------------------------- banner */

export const Banner: React.FC<StillProps> = ({mode}) => {
  const t = T[mode];
  return (
    <AbsoluteFill style={{background: t.bg, backgroundImage: t.glow, padding: '0 90px', justifyContent: 'center', fontFamily: sans}}>
      <Wordmark size={64} mode={mode} />
      <div style={{marginTop: 26, fontSize: 54, fontWeight: 800, color: t.text, lineHeight: 1.1, letterSpacing: -1}}>
        Does Timestral hold up on <span style={{color: t.accent}}>Indonesian food prices?</span>
      </div>
      <div style={{marginTop: 18, fontSize: 28, color: t.muted, fontWeight: 500}}>
        Zero-shot on a CPU laptop · 4 claims, criteria locked before running · PIHPS daily prices 2018–2026
      </div>
      <div style={{marginTop: 30, display: 'flex', gap: 14, flexWrap: 'wrap'}}>
        {MODELS.map((m, i) => (
          <Chip key={m} mode={mode} hi={i === 0}>{m}</Chip>
        ))}
      </div>
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------ diagram */

const Box: React.FC<{mode: Mode; x: number; y: number; w: number; h: number; hi?: boolean; children: React.ReactNode}> = ({
  mode, x, y, w, h, hi, children,
}) => (
  <div
    style={{
      position: 'absolute', left: x, top: y, width: w, height: h, boxSizing: 'border-box',
      background: hi ? T[mode].accentSoft : T[mode].panel,
      border: `2px solid ${hi ? T[mode].accent : T[mode].border}`,
      borderRadius: 18, padding: '22px 24px',
    }}
  >
    {children}
  </div>
);

const Label: React.FC<{mode: Mode; children: React.ReactNode}> = ({mode, children}) => (
  <div style={{fontSize: 17, fontWeight: 700, letterSpacing: 2, color: T[mode].muted, textTransform: 'uppercase'}}>{children}</div>
);

const Lines: React.FC<{mode: Mode; items: string[]; monoFont?: boolean; size?: number}> = ({mode, items, monoFont, size = 23}) => (
  <div style={{marginTop: 14, fontFamily: monoFont ? mono : sans, fontSize: size, color: T[mode].text, lineHeight: 1.6}}>
    {items.map((l) => (
      <div key={l}>{l}</div>
    ))}
  </div>
);

const Note: React.FC<{mode: Mode; children: React.ReactNode}> = ({mode, children}) => (
  <div style={{marginTop: 14, fontSize: 19, color: T[mode].muted, lineHeight: 1.4}}>{children}</div>
);

/** Pipeline: PIHPS + calendar → models → rolling-origin backtest → claims. */
export const Diagram: React.FC<StillProps> = ({mode}) => {
  const t = T[mode];
  const y = 110;
  const h = 620;
  const w = 330;
  const gap = 60;
  const xs = [50, 50 + w + gap, 50 + 2 * (w + gap), 50 + 3 * (w + gap)];
  const mid = y + h / 2;
  return (
    <AbsoluteFill style={{background: t.bg, fontFamily: sans}}>
      <svg width={1600} height={860} style={{position: 'absolute', inset: 0}}>
        <defs>
          <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill={t.accent} />
          </marker>
        </defs>
        {xs.slice(0, 3).map((x) => (
          <line key={x} x1={x + w + 4} y1={mid} x2={x + w + gap - 6} y2={mid} stroke={t.accent} strokeWidth={4} markerEnd="url(#a)" />
        ))}
      </svg>

      <div style={{position: 'absolute', left: 50, top: 40, fontSize: 30, fontWeight: 800, color: t.text}}>
        How the benchmark runs
      </div>

      <Box mode={mode} x={xs[0]} y={y} w={w} h={h}>
        <Label mode={mode}>Data</Label>
        <div style={{marginTop: 12, fontSize: 27, fontWeight: 700, color: t.text}}>PIHPS food prices</div>
        <Note mode={mode}>Bank Indonesia, 2018–2026</Note>
        <Lines mode={mode} items={['Red chili', "Bird's eye chili", 'Shallot', 'Medium rice']} />
        <div style={{marginTop: 26, fontSize: 27, fontWeight: 700, color: t.text}}>Holiday calendar</div>
        <Note mode={mode}>Known in advance, fed as future covariates</Note>
        <Lines mode={mode} items={['Ramadan · Eid al-Fitr', 'Eid al-Adha', 'Christmas · New Year']} />
      </Box>

      <Box mode={mode} x={xs[1]} y={y} w={w} h={h} hi>
        <Label mode={mode}>Models · zero-shot</Label>
        <Lines mode={mode} items={MODELS} size={25} />
        <div style={{marginTop: 22, fontSize: 21, color: t.accent, fontWeight: 700, lineHeight: 1.4}}>
          Each model that supports covariates runs with and without the calendar
        </div>
        <Note mode={mode}>Same 1,024-day history for all. Default settings, no tuning.</Note>
      </Box>

      <Box mode={mode} x={xs[2]} y={y} w={w} h={h}>
        <Label mode={mode}>Backtest</Label>
        <Lines mode={mode} items={['93 weekly origins', '× 4 commodities', '28-day forecasts']} size={25} />
        <Note mode={mode}>Jan 2025 – Oct 2026</Note>
        <div style={{marginTop: 26, fontSize: 27, fontWeight: 700, color: t.text}}>Metrics</div>
        <Lines mode={mode} items={['MASE (accuracy)', 'CRPS (probabilistic)', '80% / 90% coverage', 'CPU latency, M1']} />
      </Box>

      <Box mode={mode} x={xs[3]} y={y} w={w} h={h}>
        <Label mode={mode}>Claims tested</Label>
        <Lines mode={mode} items={['K1 · more accurate', 'K2 · calendar helps', 'K3 · small and fast', 'K4 · calibrated']} size={25} />
        <Note mode={mode}>Each ends as proven, partly proven or not proven. Rules fixed before the first run.</Note>
        <div style={{marginTop: 26, fontSize: 27, fontWeight: 700, color: t.text}}>Outputs</div>
        <Lines mode={mode} items={['results/claims.md', 'results/summary.csv', 'chart_lebaran_2026.png']} monoFont size={19} />
      </Box>

      <div style={{position: 'absolute', left: 50, top: y + h + 40, fontSize: 21, color: t.muted}}>
        Rivals are scored in their best configuration. Timestral is never given a setting the others do not get.
      </div>
    </AbsoluteFill>
  );
};
