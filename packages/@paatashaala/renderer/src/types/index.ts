// The slide object model is the canonical contract from @paatashaala/dsl. The renderer
// no longer vendors its own copy; it re-exports the DSL types here so the public
// `@paatashaala/renderer/types` surface stays intact.
export * from '@paatashaala/dsl';
export * from './effects';
