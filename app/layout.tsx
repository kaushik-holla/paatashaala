import type { Metadata } from 'next';
import { GeistSans } from 'geist/font/sans';
import { GeistMono } from 'geist/font/mono';
import './globals.css';
import '@paatashaala/renderer/fonts.css';
import 'animate.css';
import 'katex/dist/katex.min.css';
import { ThemeProvider } from '@/lib/hooks/use-theme';
import { I18nProvider } from '@/lib/hooks/use-i18n';
import { Toaster } from '@/components/ui/sonner';
import { ServerProvidersInit } from '@/components/server-providers-init';
import { StorageHealthNotice } from '@/components/storage-health-notice';
import { AccessCodeGuard } from '@/components/access-code-guard';
import { ProSwapWatcher } from '@/components/workbench/ProSwapWatcher';

// Lato and Playfair Display are assigned by the design tokens in globals.css.
// Inter remains the multilingual fallback because @fontsource's stylesheet
// carries per-subset `unicode-range` declarations. That keeps Cyrillic,
// Vietnamese and other unsupported Lato glyphs on a predictable bundled face
// instead of an arbitrary OS fallback.
//
// Declaring the other subset files as sibling faces of the same family does not
// fix it either: faces with identical descriptors and no `unicode-range` do not
// fall through per glyph, so the browser simply picks one.
//
import '@fontsource-variable/inter';

export const metadata: Metadata = {
  title: 'Paatashaala',
  description:
    'An open-source AI interactive classroom. Upload a PDF to instantly generate an immersive, multi-agent learning experience.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${GeistSans.variable} ${GeistMono.variable} antialiased`}
        suppressHydrationWarning
      >
        <ThemeProvider>
          <I18nProvider>
            <ServerProvidersInit />
            <ProSwapWatcher />
            <AccessCodeGuard>{children}</AccessCodeGuard>
            <Toaster position="top-center" />
            {/* After the Toaster: this one raises a toast on mount when
                persistence is already broken, and a toast raised before its
                host exists has nowhere to go. */}
            <StorageHealthNotice />
          </I18nProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
