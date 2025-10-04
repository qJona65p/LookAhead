import React from 'react';
import { FaceDetectionProvider } from '@infinitered/react-native-mlkit-face-detection';

export default function FaceDetectionWrapper({ children }: { children: React.ReactNode }) {
    return <FaceDetectionProvider>{ children }</FaceDetectionProvider>
}