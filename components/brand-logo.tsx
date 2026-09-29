'use client';

import type { ImgHTMLAttributes } from 'react';
import { useBrand } from '@/lib/brand/brand-context';
import { useOptionalTheme } from '@/lib/hooks/use-theme';

type BrandLogoProps = Omit<ImgHTMLAttributes<HTMLImageElement>, 'src'>;

export function BrandLogo({ alt, ...props }: BrandLogoProps) {
  const brand = useBrand();
  const theme = useOptionalTheme();

  return (
    <img
      {...props}
      src={theme?.resolvedTheme === 'dark' ? brand.logoSrcDark : brand.logoSrc}
      alt={alt ?? brand.productName}
    />
  );
}
