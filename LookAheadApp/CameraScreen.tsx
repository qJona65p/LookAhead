import { useState, useEffect, useRef, useMemo } from 'react';
import { View, StyleSheet, Text, useWindowDimensions } from 'react-native';
import { Camera as VisionCamera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { Camera, Face, FaceDetectionOptions, Landmarks, Contours } from 'react-native-vision-camera-face-detector';
import Animated, { useSharedValue, useAnimatedStyle, withTiming } from 'react-native-reanimated';
import { Canvas, Circle, Path } from '@shopify/react-native-skia';
import Skia from '@shopify/react-native-skia';
import React from 'react';

export default function CameraScreen() {
    const { width: viewWidth, height: viewHeight } = useWindowDimensions();
    const { hasPermission, requestPermission } = useCameraPermission();
    
    const [ faceStatus, setFaceStatus ] = useState<{ yaw: string; pitch: string; eye: string } | null>(null);
    const [ landmarks, setLandmarks ] = useState<Landmarks | undefined>(undefined);
    const [ contours, setContours ] = useState<Contours | undefined>(undefined);

    const device = useCameraDevice('front');

    // Dentro del componente
    const shouldMirror = device?.position === 'front';
    const mirrorX = (x: number) => (shouldMirror ? viewWidth - x : x);

    // Permission
    useEffect(() => {
        if (!hasPermission) {
            requestPermission();
        }
    }, [hasPermission, requestPermission]);

    if (hasPermission === null) {return (<View style={styles.container}><Text>Rquesting camera permission</Text></View>);}
    if (hasPermission === false) {return (<View style={styles.container}><Text>No camera access</Text></View>);}
    if (!device) {return (<View style={styles.container}><Text>No front camera available</Text></View>);}

    const aFaceW = useSharedValue(0);
    const aFaceH = useSharedValue(0);
    const aFaceX = useSharedValue(0);
    const aFaceY = useSharedValue(0);

    const drawFaceBounds = (face?: Face) => {
        if (face) {
            const { width, height, x, y } = face.bounds;
            aFaceW.value = width;
            aFaceH.value = height;
            aFaceX.value = x;
            aFaceY.value = y;
        } else {
            aFaceW.value = aFaceH.value = aFaceX.value = aFaceY.value = 0;
        }
    }

    const faceBoxStyle = useAnimatedStyle(() => ({
        position: 'absolute',
        borderWidth: 4,
        borderColor: 'rgb(0,255,0)',
        width: withTiming(aFaceW.value, { duration: 100 }),
        height: withTiming(aFaceH.value, { duration: 100 }),
        left: withTiming(aFaceX.value, { duration: 100 }),
        top: withTiming(aFaceY.value, { duration: 100 })
    }));

    const faceDetectionOptions = useRef<FaceDetectionOptions>({
        performanceMode: 'fast',
        landmarkMode: 'all',
        contourMode: 'all',
        classificationMode: 'all',
        trackingEnabled: false,
        windowWidth: viewWidth,
        windowHeight: viewHeight,
        autoMode: true,
    }).current;

    const handleFacesDetection = (faces: Face[]) => {
        try {
            if (faces?.length > 0) {
                const face = faces[0];

                drawFaceBounds(face);
                setLandmarks(face.landmarks);
                setContours(face.contours);

                setFaceStatus({
                    yaw: face.yawAngle > 15 ? "Right" : face.yawAngle < -15 ? "Left" : "Center",
                    pitch: face.pitchAngle > 15 ? "Up" : face.pitchAngle < -10 ? "Down" : "Center",
                    eye: face.leftEyeOpenProbability > 0.7 && face.rightEyeOpenProbability > 0.7 ? "Open" : "Close"
                });
            } else {
                drawFaceBounds();
                setLandmarks(undefined);
                setContours(undefined);
                setFaceStatus(null);
            }
        } catch (error) {
            console.error("Error in face detection:", error);
        }
    }

    const memoizedLandmarks = useMemo(() => {
        if (!landmarks) return null;

        const safeCircle = (
            point: { x: number; y: number } | undefined,
            r: number,
            color: string
        ) => {
            if (!point) return null;
            return <Circle cx={mirrorX(point.x)} cy={point.y} r={r} color={color} />;
        };

        return (
            <>
                {/* Ojos */}
                {safeCircle(landmarks.LEFT_EYE, 8, "cyan")}
                {safeCircle(landmarks.RIGHT_EYE, 8, "cyan")}

                {/* Nariz */}
                {safeCircle(landmarks.NOSE_BASE, 7, "yellow")}

                {/* Boca */}
                {safeCircle(landmarks.MOUTH_LEFT, 6, "red")}
                {safeCircle(landmarks.MOUTH_RIGHT, 6, "red")}

                {/* Orejas */}
                {safeCircle(landmarks.LEFT_EAR, 7, "magenta")}
                {safeCircle(landmarks.RIGHT_EAR, 7, "magenta")}

                {/* Mejillas */}
                {safeCircle(landmarks.LEFT_CHEEK, 6, "lightblue")}
                {safeCircle(landmarks.RIGHT_CHEEK, 6, "lightblue")}
            </>
        );
    }, [landmarks, viewWidth, shouldMirror]);

    return (
        <View style={styles.container}>
            <Camera
                style={styles.camera}
                device={device}
                isActive={true}
                faceDetectionCallback={handleFacesDetection}
                faceDetectionOptions={faceDetectionOptions}
                onError={(error) => console.log('Camera error:', error)}
            />
            <Animated.View style={[faceBoxStyle, styles.animatedView]}>
                <Text style={styles.statusText}>Yaw: {faceStatus?.yaw}</Text>
                <Text style={styles.statusText}>Pitch: {faceStatus?.pitch}</Text>
                <Text style={styles.statusText}>Eye: {faceStatus?.eye}</Text>
            </Animated.View>
            <Canvas style={StyleSheet.absoluteFill}>
                {memoizedLandmarks}
            </Canvas>
        </View>
    )
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center'
    },
    camera: {
        flex: 1
    },
    animatedView: {
        justifyContent: 'flex-end',
        alignItems: 'flex-start',
        borderRadius: 20,
        padding: 10,
    },
    statusText: {
        color: 'lightgreen',
        fontSize: 14,
        fontWeight: 'bold',
    },
})