'use client';

import { useEffect, useMemo, useRef } from 'react';
import { AlertTriangle, ShieldAlert } from 'lucide-react';

import {
  MalpracticeFlag,
  MalpracticeFlagType,
  useMalpracticeMonitor,
} from '@/hooks/useMalpracticeMonitor';
import { useToast } from '@/hooks/use-toast';

const formatDuration = (since: number) => {
  const elapsedMs = Date.now() - since;
  if (elapsedMs < 1_000) return 'just now';
  if (elapsedMs < 60_000) {
    const seconds = Math.round(elapsedMs / 1_000);
    return `${seconds}s ago`;
  }
  const minutes = Math.round(elapsedMs / 60_000);
  return `${minutes}m ago`;
};

const MalpracticeWarningList = ({ flags }: { flags: MalpracticeFlag[] }) => {
  if (!flags.length) return null;

  return (
    <ul className="mt-2 space-y-1">
      {flags.map((flag) => (
        <li key={flag.type} className="flex flex-col">
          <span className="text-sm font-medium leading-tight">
            {flag.message}
          </span>
          <span className="text-xs text-white/70">
            Flagged {formatDuration(flag.since)}
          </span>
        </li>
      ))}
    </ul>
  );
};

const MalpracticeSensor = () => {
  const { flags, error, isMonitoring, detectorAvailable, detectorSource } =
    useMalpracticeMonitor();
  const { toast } = useToast();
  const previousFlagsRef = useRef<Set<MalpracticeFlagType>>(new Set());
  const hasWarnings = flags.length > 0;

  useEffect(() => {
    const previous = previousFlagsRef.current;
    const next = new Set(flags.map((flag) => flag.type));

    flags.forEach((flag) => {
      if (!previous.has(flag.type)) {
        toast({
          title: 'Malpractice warning',
          description: flag.message,
        });
      }
    });

    previousFlagsRef.current = next;
  }, [flags, toast]);

  useEffect(() => {
    if (error) {
      toast({
        title: 'Malpractice sensor disabled',
        description: error,
      });
    }
  }, [error, toast]);

  const containerStyles = useMemo(() => {
    if (hasWarnings) {
      return 'pointer-events-none absolute right-6 top-6 z-50 max-w-sm';
    }
    if (error) {
      return 'pointer-events-auto absolute right-6 top-6 z-50 max-w-sm';
    }
    return 'pointer-events-none absolute right-6 top-6 z-50';
  }, [error, hasWarnings]);

  if (!detectorAvailable) {
    return (
      <div className={containerStyles}>
        <div className="rounded-lg bg-amber-500/90 px-4 py-3 text-sm font-medium text-white shadow-lg backdrop-blur">
          <div className="flex items-center gap-2">
            <ShieldAlert className="h-4 w-4" />
            <span>Malpractice sensor unavailable</span>
          </div>
          <p className="mt-1 text-xs text-white/80">
            {error ??
              'Update your browser or refresh the page to enable proctoring safeguards.'}
          </p>
        </div>
      </div>
    );
  }

  if (!hasWarnings && !error) {
    return null;
  }

  return (
    <div className={containerStyles}>
      <div className="rounded-lg bg-red-500/90 px-4 py-3 text-white shadow-lg backdrop-blur">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4" />
          <span className="text-sm font-semibold">Malpractice Detected</span>
        </div>
        <MalpracticeWarningList flags={flags} />
        {detectorSource === 'fallback' && !error && (
          <p className="mt-2 text-xs text-white/70">
            Face detection running in compatibility mode.
          </p>
        )}
        {!flags.length && error && (
          <p className="mt-2 text-sm font-medium leading-tight text-white">
            {error}
          </p>
        )}
        {!isMonitoring && !error && (
          <p className="mt-2 text-xs text-white/80">
            Monitoring paused. Ensure camera access is granted.
          </p>
        )}
      </div>
    </div>
  );
};

export default MalpracticeSensor;

