interface FaceDetection {
  boundingBox: DOMRectReadOnly;
  landmarks?: { locations: DOMPointReadOnly[] }[];
}

interface FaceDetectorOptions {
  maxDetectedFaces?: number;
  fastMode?: boolean;
}

interface FaceDetector {
  detect: (image: CanvasImageSource) => Promise<FaceDetection[]>;
}

interface Window {
  FaceDetector?: {
    new (options?: FaceDetectorOptions): FaceDetector;
  };
}

