import {loadFont as loadInter} from '@remotion/google-fonts/Inter';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';

export const {fontFamily: sans} = loadInter('normal', {weights: ['400', '500', '600', '700', '800'], subsets: ['latin']});
export const {fontFamily: mono} = loadMono('normal', {weights: ['400', '500', '700'], subsets: ['latin']});
