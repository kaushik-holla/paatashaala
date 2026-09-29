import { describe, expect, it } from 'vitest';
import { DEFAULT_BRAND } from '@/lib/brand/brand-config';

describe('DEFAULT_BRAND (single-brand build)', () => {
  it('uses Paatashaala identity throughout the interface', () => {
    expect(DEFAULT_BRAND.productName).toBe('Paatashaala');
    expect(DEFAULT_BRAND.shortName).toBe('Paatashaala');
    expect(DEFAULT_BRAND.markSrc).toBe('/paatashaala-mark.svg');
    expect(DEFAULT_BRAND.themeColor).toBe('#456477');
  });

  it('marks its horizontal logo as already containing the wordmark', () => {
    expect(DEFAULT_BRAND.logoHasWordmark).toBe(true);
    expect(DEFAULT_BRAND.logoSrc).toBe('/paatashaala-logo.svg');
    expect(DEFAULT_BRAND.logoSrcDark).toBe('/paatashaala-logo-light.svg');
  });
});
