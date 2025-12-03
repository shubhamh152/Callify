'use client';

import { useCall } from '@stream-io/video-react-sdk';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Subscription } from 'rxjs';

export type MalpracticeFlagType =
  | 'multiple-faces-detected'
  | 'no-face-detected'
  | 'face-near-frame-edge';

export interface MalpracticeFlag {
  type: MalpracticeFlagType;
  message: string;
  since: number;
}

interface MalpracticeState {
  flags: MalpracticeFlag[];
  error?: string;
  isMonitoring: boolean;
  detectorAvailable: boolean;
  detectorSource?: DetectorSource;
}

type DetectorSource = 'native' | 'fallback';

interface DetectionResult {
  boundingBox: DOMRectReadOnly;
}

interface DetectionAdapter {
  detect: (video: HTMLVideoElement) => Promise<DetectionResult[]>;
  dispose?: () => Promise<void> | void;
  source: DetectorSource;
}

interface TfModules {
  faceDetection: typeof import('@tensorflow-models/face-detection');
  tfCore: typeof import('@tensorflow/tfjs-core');
}

interface TfFaceBox {
  xMin: number;
  yMin: number;
  xMax?: number;
  yMax?: number;
  width: number;
  height: number;
}

const DETECTION_INTERVAL_MS = 750;
const NO_FACE_THRESHOLD_MS = 3_000;
const EDGE_THRESHOLD_MS = 1_500;
const EDGE_MARGIN_RATIO = 0.08;

const FLAG_MESSAGES: Record<MalpracticeFlagType, string> = {
  'multiple-faces-detected':
    'Multiple faces detected in frame. Possible impersonation.',
  'no-face-detected':
    'No face detected for several seconds. Head may be out of frame.',
  'face-near-frame-edge':
    'Face detected at the edge of the frame. Adjust position.',
};

const FLAG_PRIORITY: MalpracticeFlagType[] = [
  'multiple-faces-detected',
  'no-face-detected',
  'face-near-frame-edge',
];

const nativeDetectorSupported = () =>
  typeof window !== 'undefined' &&
  'FaceDetector' in window &&
  typeof window.FaceDetector === 'function';

let tfModulesPromise: Promise<TfModules> | null = null;

const ensureTfModules = async (): Promise<TfModules> => {
  if (!tfModulesPromise) {
    tfModulesPromise = (async () => {
      try {
        const tfCore = await import('@tensorflow/tfjs-core');
        await Promise.all([
          import('@tensorflow/tfjs-converter'),
          import('@tensorflow/tfjs-backend-webgl'),
          import('@tensorflow/tfjs-backend-cpu'),
        ]);

        const candidateBackends: Array<'webgl' | 'cpu'> = ['webgl', 'cpu'];
        let backendInitialized: 'webgl' | 'cpu' | null = null;

        for (const candidate of candidateBackends) {
          try {
            await tfCore.setBackend(candidate);
            await tfCore.ready();
            backendInitialized = candidate;
            break;
          } catch (error) {
            console.warn(
              `[MalpracticeMonitor] Failed to initialize TensorFlow backend "${candidate}"`,
              error,
            );
          }
        }

        if (!backendInitialized) {
          throw new Error('No TensorFlow backend available');
        }

        const faceDetection = await import('@tensorflow-models/face-detection');
        return { faceDetection, tfCore };
      } catch (error) {
        tfModulesPromise = null;
        throw error;
      }
    })();
  }

  return tfModulesPromise;
};

const createBoundingBox = (
  box: TfFaceBox,
  video: HTMLVideoElement,
): DOMRectReadOnly => {
  const { videoWidth, videoHeight } = video;
  const normalized =
    videoWidth > 0 &&
    videoHeight > 0 &&
    (box.xMax ?? 2) <= 1 &&
    (box.yMax ?? 2) <= 1;

  const toPixels = (value: number, dimension: number) =>
    normalized ? value * dimension : value;

  const x = toPixels(box.xMin, videoWidth);
  const y = toPixels(box.yMin, videoHeight);
  const width = toPixels(box.width, videoWidth);
  const height = toPixels(box.height, videoHeight);

  if (typeof DOMRectReadOnly === 'function') {
    return new DOMRectReadOnly(x, y, width, height);
  }

  return {
    x,
    y,
    width,
    height,
    top: y,
    left: x,
    right: x + width,
    bottom: y + height,
  } as DOMRectReadOnly;
};

const createNativeAdapter = (): DetectionAdapter => {
  const NativeDetector = window.FaceDetector;
  if (!NativeDetector) {
    throw new Error('FaceDetector API unavailable on this browser');
  }

  const detector = new NativeDetector({
    fastMode: true,
    maxDetectedFaces: 5,
  });

  return {
    source: 'native',
    detect: async (video) => {
      const faces = await detector.detect(video);
      return faces.map((face) => ({ boundingBox: face.boundingBox }));
    },
  };
};

const createFallbackAdapter = async (): Promise<DetectionAdapter> => {
  const { faceDetection } = await ensureTfModules();
  const model = faceDetection.SupportedModels.MediaPipeFaceDetector;

  let detector = await faceDetection
    .createDetector(model, {
      runtime: 'tfjs',
      maxFaces: 5,
    })
    .catch(async (error: unknown) => {
      console.warn(
        '[MalpracticeMonitor] TFJS runtime failed, falling back to MediaPipe runtime',
        error,
      );

      return faceDetection.createDetector(model, {
        runtime: 'mediapipe',
        maxFaces: 5,
        solutionPath:
          'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.4',
      });
    });

  if (!detector) {
    throw new Error('Unable to initialize fallback face detector');
  }

  return {
    source: 'fallback',
    detect: async (video) => {
      const faces = await detector.estimateFaces(video, {
        flipHorizontal: false,
      });

      return (faces as Array<{ box: TfFaceBox }>).map((face) => ({
        boundingBox: createBoundingBox(face.box, video),
      }));
    },
    dispose: () => detector.dispose(),
  };
};

const resolveDetector = async (): Promise<DetectionAdapter> => {
  if (nativeDetectorSupported()) {
    return createNativeAdapter();
  }
  return createFallbackAdapter();
};

const logDetectionError = (err: unknown) => {
  console.error('[MalpracticeMonitor] face detection failed', err);
};

export const useMalpracticeMonitor = (): MalpracticeState => {
  const call = useCall();

  const [flags, setFlags] = useState<MalpracticeFlag[]>([]);
  const [error, setError] = useState<string>();
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [detectorAvailable, setDetectorAvailable] = useState(true);
  const [detectorSource, setDetectorSource] = useState<DetectorSource>();

  const detectorRef = useRef<DetectionAdapter>();
  const videoRef = useRef<HTMLVideoElement>();
  const detectionTimerRef = useRef<number>();
  const mediaStreamSubRef = useRef<Subscription>();
  const noFaceSinceRef = useRef<number | null>(null);
  const edgeSinceRef = useRef<number | null>(null);

  const stopLoop = useCallback(() => {
    if (detectionTimerRef.current) {
      window.clearTimeout(detectionTimerRef.current);
      detectionTimerRef.current = undefined;
    }

    const adapter = detectorRef.current;
    detectorRef.current = undefined;
    if (adapter?.dispose) {
      void Promise.resolve(adapter.dispose()).catch((err) =>
        console.warn('[MalpracticeMonitor] failed to dispose detector', err),
      );
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    videoRef.current.remove();
    videoRef.current = undefined;
    }

    mediaStreamSubRef.current?.unsubscribe();
    mediaStreamSubRef.current = undefined;
    noFaceSinceRef.current = null;
    edgeSinceRef.current = null;
  }, []);

  const updateFlags = useCallback(
    (types: MalpracticeFlagType[], timestamp: number) => {
      setFlags((prev) => {
        const previousByType = new Map(prev.map((flag) => [flag.type, flag]));
        const prioritizedTypes = FLAG_PRIORITY.filter((type) =>
          types.includes(type),
        );

        if (
          prioritizedTypes.length === prev.length &&
          prioritizedTypes.every((type, index) => prev[index]?.type === type)
        ) {
          return prev;
        }

        return prioritizedTypes.map((type) => {
          const existing = previousByType.get(type);
          return (
            existing ?? {
              type,
              message: FLAG_MESSAGES[type],
              since: timestamp,
            }
          );
        });
      });
    },
    [],
  );

  const resetFlags = useCallback(() => {
    setFlags((prev) => (prev.length ? [] : prev));
    noFaceSinceRef.current = null;
    edgeSinceRef.current = null;
  }, []);

  useEffect(() => {
    if (!call || typeof window === 'undefined') {
      return () => undefined;
    }

    let isCancelled = false;
    const video = document.createElement('video');
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    video.style.position = 'fixed';
    video.style.opacity = '0';
    video.style.pointerEvents = 'none';
    document.body.appendChild(video);

    videoRef.current = video;
    setDetectorAvailable(true);
    setError(undefined);

    const attachStream = (stream?: MediaStream) => {
      if (!videoRef.current) return;

      if (stream) {
        if (videoRef.current.srcObject !== stream) {
          videoRef.current.srcObject = stream;
          void videoRef.current.play().catch(() => {
            /* playback errors can be safely ignored */
          });
        }
      } else {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
      }
    };

    const initialize = async () => {
      try {
        const adapter = await resolveDetector();

        if (isCancelled) {
          adapter.dispose?.();
          return;
        }

        detectorRef.current = adapter;
        setDetectorSource(adapter.source);
        setDetectorAvailable(true);
        setIsMonitoring(true);
      } catch (err) {
        if (!isCancelled) {
          logDetectionError(err);
          setDetectorAvailable(false);
          setDetectorSource(undefined);
          setError(
            'Face detection libraries failed to load. Malpractice monitoring is unavailable.',
          );
          setIsMonitoring(false);
          resetFlags();
        }
        return;
      }

      attachStream(call.camera.state.mediaStream ?? undefined);

      if (call.camera.state.mediaStream$) {
        mediaStreamSubRef.current = call.camera.state.mediaStream$.subscribe(
          (stream) => {
            attachStream(stream ?? undefined);
          },
        );
      }

      const runDetection = async () => {
        if (isCancelled || !detectorRef.current || !videoRef.current) {
          return;
        }

        if (!videoRef.current.srcObject) {
          resetFlags();
          detectionTimerRef.current = window.setTimeout(
            runDetection,
            DETECTION_INTERVAL_MS,
          );
          return;
        }

        if (videoRef.current.readyState < 2 || !videoRef.current.videoWidth) {
          detectionTimerRef.current = window.setTimeout(
            runDetection,
            DETECTION_INTERVAL_MS,
          );
          return;
        }

        try {
          const faces = await detectorRef.current.detect(videoRef.current);
          const orderedFaces =
            faces.length > 1
              ? [...faces].sort((a, b) => {
                  const areaA =
                    a.boundingBox.width * a.boundingBox.height;
                  const areaB =
                    b.boundingBox.width * b.boundingBox.height;
                  return areaB - areaA;
                })
              : faces;
          const now = Date.now();
          const nextFlags: MalpracticeFlagType[] = [];

          if (orderedFaces.length === 0) {
            if (noFaceSinceRef.current === null) {
              noFaceSinceRef.current = now;
            }
            if (now - (noFaceSinceRef.current ?? now) >= NO_FACE_THRESHOLD_MS) {
              nextFlags.push('no-face-detected');
            }
            edgeSinceRef.current = null;
          } else {
            noFaceSinceRef.current = null;

            if (orderedFaces.length > 1) {
              nextFlags.push('multiple-faces-detected');
            }

            const primaryFace = orderedFaces[0];
            const { boundingBox } = primaryFace;
            const { videoWidth, videoHeight } = videoRef.current;

            if (videoWidth && videoHeight) {
              const left = boundingBox.x / videoWidth;
              const top = boundingBox.y / videoHeight;
              const right = (boundingBox.x + boundingBox.width) / videoWidth;
              const bottom = (boundingBox.y + boundingBox.height) / videoHeight;

              const touchesEdge =
                left <= EDGE_MARGIN_RATIO ||
                top <= EDGE_MARGIN_RATIO ||
                right >= 1 - EDGE_MARGIN_RATIO ||
                bottom >= 1 - EDGE_MARGIN_RATIO;

              if (touchesEdge) {
                if (edgeSinceRef.current === null) {
                  edgeSinceRef.current = now;
                }
                if (
                  now - (edgeSinceRef.current ?? now) >= EDGE_THRESHOLD_MS &&
                  !nextFlags.includes('multiple-faces-detected')
                ) {
                  nextFlags.push('face-near-frame-edge');
                }
              } else {
                edgeSinceRef.current = null;
              }
            }
          }

          if (nextFlags.length) {
            updateFlags(nextFlags, now);
          } else {
            resetFlags();
          }
        } catch (err) {
          if (!isCancelled) {
            logDetectionError(err);
            setError(
              err instanceof Error
                ? err.message
                : 'Face detection failed unexpectedly.',
            );
            setIsMonitoring(false);
            stopLoop();
            setDetectorAvailable(false);
            setDetectorSource(undefined);
            resetFlags();
            return;
          }
        }

        detectionTimerRef.current = window.setTimeout(
          runDetection,
          DETECTION_INTERVAL_MS,
        );
      };

      detectionTimerRef.current = window.setTimeout(
        runDetection,
        DETECTION_INTERVAL_MS,
      );
    };

    void initialize();

    return () => {
      isCancelled = true;
      stopLoop();
      video.remove();
      resetFlags();
      setIsMonitoring(false);
      setDetectorSource(undefined);
    };
  }, [call, resetFlags, stopLoop, updateFlags]);

  useEffect(() => {
    return () => {
      stopLoop();
      setIsMonitoring(false);
      resetFlags();
      setDetectorSource(undefined);
    };
  }, [resetFlags, stopLoop]);

  return { flags, error, isMonitoring, detectorAvailable, detectorSource };
};

