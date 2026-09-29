'use client';

import { Button } from '@/components/ui/button';
import { Box, Plus } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useI18n } from '@/lib/hooks/use-i18n';
import type { ProviderId, ProviderConfig } from '@/lib/ai/providers';
import { MONO_LOGO_PROVIDERS } from '@/lib/ai/providers';

interface ProviderWithServerInfo extends ProviderConfig {
  isServerConfigured?: boolean;
}

interface ProviderListProps {
  providers: ProviderWithServerInfo[];
  selectedProviderId: ProviderId;
  onSelect: (providerId: ProviderId) => void;
  onAddProvider: () => void;
  width?: number;
}

export function ProviderList({
  providers,
  selectedProviderId,
  onSelect,
  onAddProvider,
  width,
}: ProviderListProps) {
  const { t } = useI18n();

  // Helper function to get translated provider name
  const getProviderDisplayName = (provider: ProviderConfig) => {
    const translationKey = `settings.providerNames.${provider.id}`;
    const translated = t(translationKey);
    // If translation exists (not equal to key), use it; otherwise fallback to provider.name
    return translated !== translationKey ? translated : provider.name;
  };

  return (
    <div
      className="flex-shrink-0 bg-background flex flex-col max-sm:!w-full max-sm:!flex-row max-sm:h-[72px] max-sm:border-b"
      style={{ width: width ?? 192 }}
    >
      <div className="settings-horizontal-scroll flex-1 overflow-y-auto p-3 space-y-1.5 max-sm:flex max-sm:overflow-x-auto max-sm:overflow-y-hidden max-sm:space-y-0 max-sm:gap-1 max-sm:p-2">
        {providers.map((provider) => (
          <button
            key={provider.id}
            onClick={() => onSelect(provider.id)}
            className={cn(
              'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg transition-all border text-left max-sm:w-auto max-sm:shrink-0 max-sm:px-2',
              selectedProviderId === provider.id
                ? 'bg-primary/5 border-primary/50 shadow-sm'
                : 'border-transparent hover:bg-muted/50',
            )}
          >
            {provider.icon ? (
              <img
                src={provider.icon}
                alt={getProviderDisplayName(provider)}
                className={cn(
                  'w-5 h-5 rounded',
                  MONO_LOGO_PROVIDERS.has(provider.id) && 'dark:invert',
                )}
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
            ) : (
              <Box className="h-5 w-5 text-muted-foreground" />
            )}
            <span className="font-medium text-sm flex-1 truncate">
              {getProviderDisplayName(provider)}
            </span>
            {provider.isServerConfigured && (
              <span className="text-[10px] px-1 py-0 h-4 leading-4 rounded shrink-0 bg-muted text-muted-foreground">
                {t('settings.serverConfigured')}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Add Provider Button */}
      <div className="p-3 border-t max-sm:shrink-0 max-sm:border-t-0 max-sm:p-2">
        <Button
          variant="outline"
          size="sm"
          className="w-full gap-1.5 max-sm:w-auto"
          onClick={onAddProvider}
        >
          <Plus className="h-3.5 w-3.5" />
          {t('settings.addProviderButton')}
        </Button>
      </div>
    </div>
  );
}
