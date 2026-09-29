# @paatashaala/renderer

Internal React component for rendering PPTist-style `Slide` JSON in [Paatashaala](https://github.com/kaushik-holla/paatashaala).

> `@paatashaala/renderer` is the read-only canvas package. Editing lives in the
> separate `@paatashaala/editor` package, which depends on this renderer.

## Migrating from 0.0.x

Version `0.1.0` removes the experimental `@paatashaala/renderer/editing` subpath.
Install `@paatashaala/editor` and migrate editing imports to its explicit layers:

```ts
import { EditableSlideCanvas } from '@paatashaala/editor/react';
import { EditableSlideCanvasWithUI } from '@paatashaala/editor/ui';
```

Read-only rendering imports from `@paatashaala/renderer` are unchanged.

## Workspace setup

This package is private and used by the Paatashaala workspace. Install dependencies from the
repository root with `pnpm install`, then run `pnpm build:packages`. It is not published to npm.

Required peers:

- `react >= 18`
- `react-dom >= 18`
- `motion >= 11`
- `tailwindcss >= 4` — **the package emits Tailwind 4 arbitrary-value classes, consumers must use Tailwind 4**

Optional peers (install only if your slides use the corresponding element type):

- `echarts >= 5` — for chart elements
- `shiki >= 1` — for code elements

## Quickstart

```tsx
import { SlideCanvas, type Slide } from '@paatashaala/renderer';

const slide: Slide = {
  id: 'demo-1',
  viewportSize: 1000,
  viewportRatio: 0.5625,
  theme: {
    backgroundColor: '#ffffff',
    themeColors: ['#5b8def'],
    fontColor: '#222222',
    fontName: 'sans-serif',
  },
  elements: [
    {
      type: 'text',
      id: 't1',
      left: 100,
      top: 80,
      width: 800,
      height: 60,
      rotate: 0,
      content: '<p>Hello, Slide</p>',
      defaultFontName: 'sans-serif',
      defaultColor: '#222',
    },
  ],
  background: { type: 'solid', color: '#ffffff' },
};

export default function Demo() {
  return (
    <div style={{ width: 800, height: 450 }}>
      <SlideCanvas slide={slide} />
    </div>
  );
}
```

The canvas auto-fits its parent container. The parent must have a defined `width × height`.

## API

### `<SlideCanvas slide effects? renderImage? renderVideo? onElementClick? scale? background? />`

The main read-only entry. Reads everything from props; zero global state.

```ts
interface SlideCanvasProps {
  slide?: Slide;                       // required unless via <SlideRendererProvider>
  scale?: number;                      // omit = auto-fit container
  canvasPercentage?: number;           // percent of the parent used by auto-fit
  onScaleChange?: (scale) => void;     // computed auto-fit scale
  background?: SlideBackground;        // overrides slide.background
  effects?: SlideEffects;              // laser / spotlight / highlight / zoom, all default off
  renderImage?: (el, src, defaultContent) => ReactNode;
  renderVideo?: (el) => ReactNode;
  videoInteractive?: boolean;           // defaults true; set false to disable video pointer interaction
  onElementClick?: (el, event) => void;
  elementIdPrefix?: string;
  className?: string;
  style?: CSSProperties;
}
```

### Play-time effects

All effects are off by default. Pass any combination via `effects`:

```tsx
<SlideCanvas
  slide={slide}
  effects={{
    laser:     { elementId: 't1', color: '#ff3b30' },
    spotlight: { elementId: 't1' },
    highlight: { elementId: 't1', color: '#ff6b6b', animated: true },
    zoom:      { elementId: 't1', scale: 1.5 },
  }}
/>
```

### Media injection slots

The package's `BaseImageElement` and `BaseVideoElement` render plain `<img>` / `<video>` and know nothing about your media pipeline. Inject business behaviour via the `renderImage` / `renderVideo` slots:

```tsx
<SlideCanvas
  slide={slide}
  renderImage={(el, src, defaultContent) => (
    src.startsWith('placeholder:')
      ? <MyPlaceholder taskId={src} />
      : defaultContent
  )}
/>
```

`defaultContent` is the renderer-prepared image, including clipping, filters,
soft edges, and `colorMask`. The slot return value is authoritative: return
`null` to intentionally hide the image.

### `<SlideRendererProvider>` + `useSlideContext()`

Optional high-order pattern when sibling overlays need the same slide data:

```tsx
import { SlideRendererProvider, SlideCanvas, useSlideContext } from '@paatashaala/renderer';

function MyAnnotationLayer() {
  const { slide } = useSlideContext();
  return <div>Annotations for {slide.id}</div>;
}

<SlideRendererProvider slide={slide} scale={0.9}>
  <SlideCanvas /> {/* reads slide/scale from context */}
  <MyAnnotationLayer />
</SlideRendererProvider>
```

### Granular components — `@paatashaala/renderer/elements`

If you want to compose your own layout instead of using `SlideCanvas`, the 9 base elements are exported individually:

```ts
import {
  BaseTextElement, BaseShapeElement, BaseImageElement,
  BaseLineElement, BaseChartElement, BaseLatexElement,
  BaseTableElement, BaseVideoElement, BaseCodeElement,
  ElementOutline,
} from '@paatashaala/renderer/elements';
```

Each accepts `{ elementInfo: PPTXxxElement }`. Image/Video also take a render slot.

### Types — `@paatashaala/renderer/types`

```ts
import type {
  Slide, PPTElement, SlideBackground, SlideTheme,
  PPTTextElement, PPTShapeElement, PPTImageElement,
  PPTLineElement, PPTChartElement, PPTLatexElement,
  PPTTableElement, PPTVideoElement, PPTCodeElement,
  ImageElementClip, ImageElementFilters,
  Gradient, GradientType, PPTElementOutline, PPTElementShadow,
  SlideEffects, LaserEffectOptions, SpotlightEffectOptions,
  HighlightEffectOptions, ZoomEffectOptions,
} from '@paatashaala/renderer/types';
```

## Tailwind 4 setup

Ensure your `tailwind.config.{ts,js}` includes the package source:

```js
export default {
  content: [
    './src/**/*.{ts,tsx}',
    './node_modules/@paatashaala/renderer/dist/**/*.{js,cjs}',
  ],
};
```

## Fonts (optional, local-only)

Slides imported from PowerPoint can reference faces absent from the viewer's
machine. The package ships a `fonts.css` with local-only aliases for common
imported names. Import it once at your app shell:

```ts
import '@paatashaala/renderer/fonts.css';
```

The aliases load only fonts already installed on the device; no request goes to
an external font host. If a face is unavailable, the browser uses its normal
fallback. See [FONTS.md](./FONTS.md) for the aliases and retained license notices.

## Companion package

[`@paatashaala/importer`](../importer) converts `.pptx` files to the same `Slide[]` shape, so you can do `.pptx → @paatashaala/renderer` end-to-end.

## See also

- [DESIGN.md](./DESIGN.md) — package design decisions and scope
- v2 will add editing (`<SlideEditor editable onChange />`); the read-only `<SlideCanvas>` API will remain stable

## License

MIT
