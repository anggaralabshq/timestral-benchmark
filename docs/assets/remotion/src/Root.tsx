import React from 'react';
import {Composition} from 'remotion';
import {Banner, Diagram} from './Stills';

export const Root: React.FC = () => (
  <>
    <Composition id="Banner" component={Banner} width={1600} height={440} fps={30} durationInFrames={1} defaultProps={{mode: 'dark' as const}} />
    <Composition id="Diagram" component={Diagram} width={1600} height={860} fps={30} durationInFrames={1} defaultProps={{mode: 'dark' as const}} />
  </>
);
