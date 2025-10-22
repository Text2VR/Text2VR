/// <reference types="aframe" />

declare global {
  namespace JSX {
    interface IntrinsicElements {
      'a-scene': any;
      'a-sky': any;
      'a-entity': any;
      'a-camera': any;
      'a-light': any;
    }
  }
}