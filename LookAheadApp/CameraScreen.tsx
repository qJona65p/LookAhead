import React, { useState, useEffect, useRef, useMemo } from 'react';
import { View, StyleSheet, Text, useWindowDimensions, TouchableOpacity } from 'react-native';
import { useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { Camera, Face, FaceDetectionOptions, Landmarks } from 'react-native-vision-camera-face-detector';
import { useSharedValue, useAnimatedStyle, withTiming } from 'react-native-reanimated';
import { Canvas, Circle, Image, useImage, Path } from '@shopify/react-native-skia';

const HAT_PNG = require('./assets/bowler_hat.png');
const GLASSES_PNG = require('./assets/sunglasses.png');
const MUSTACHE_PNG = require('./assets/mustache.png');

type FilterType = 'none' | 'hat' | 'glasses' | 'mustache';

export default function CameraScreen() {
    const { width: viewWidth, height: viewHeight } = useWindowDimensions();
    const { hasPermission, requestPermission } = useCameraPermission();
    
    const [ faceStatus, setFaceStatus ] = useState<{ yaw: string; pitch: string; eye: string } | null>(null);
    const [ landmarks, setLandmarks ] = useState<Landmarks | undefined>(undefined);
    const [filter, setFilter] = useState<FilterType>('none');

    const device = useCameraDevice('front');
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

    const faceDetectionOptions = useRef<FaceDetectionOptions>({
        performanceMode: 'fast',
        landmarkMode: 'all',
        classificationMode: 'all',
        trackingEnabled: false,
        windowWidth: viewWidth * 1.5,
        windowHeight: viewHeight * 1.2,
        autoMode: true,
    }).current;

    const handleFacesDetection = (faces: Face[]) => {
        try {
            if (faces?.length > 0) {
                const face = faces[0];

                drawFaceBounds(face);
                setLandmarks(face.landmarks);

                setFaceStatus({
                    yaw: face.yawAngle > 15 ? "Right" : face.yawAngle < -15 ? "Left" : "Center",
                    pitch: face.pitchAngle > 15 ? "Up" : face.pitchAngle < -10 ? "Down" : "Center",
                    eye: face.leftEyeOpenProbability > 0.7 && face.rightEyeOpenProbability > 0.7 ? "Open" : "Close"
                });
            } else {
                drawFaceBounds();
                setLandmarks(undefined);
                setFaceStatus(null);
            }
        } catch (error) {
            console.error("Error in face detection:", error);
        }
    }

    // Load images
    const hatImage = useImage(HAT_PNG);
    const glassesImage = useImage(GLASSES_PNG);
    const mustacheImage = useImage(MUSTACHE_PNG);

    const memoizedFilter = useMemo(() => {
        if (!landmarks || !landmarks.LEFT_EYE || !landmarks.RIGHT_EYE) return null;

        const leftEye = landmarks.LEFT_EYE;
        const rightEye = landmarks.RIGHT_EYE;
        const nose = landmarks.NOSE_BASE;

        const eyeMidX = (leftEye.x + rightEye.x) / 2;
        const eyeMidY = (leftEye.y + rightEye.y) / 2;
        const eyeDistance = Math.abs(rightEye.x - leftEye.x);

        const faceWidth = eyeDistance * 3.5;
        const faceHeight = faceWidth * 1.3;

        const scale = faceWidth / 300;

        const mirroredX = (x: number) => mirrorX(x);

        // Hat
        if (filter === 'hat' && hatImage) {
            const hatWidth = faceWidth;
            const hatHeight = hatWidth;
            const hatX = mirroredX(eyeMidX + hatWidth / 4.5);
            const hatY = eyeMidY - faceHeight;

            return (
                <Image
                    image={hatImage}
                    x={hatX}
                    y={hatY}
                    width={hatWidth}
                    height={hatHeight}
                    fit="contain"
                />
            );
        }

        // Glasses
        if (filter === 'glasses' && glassesImage) {
            const glassesWidth = eyeDistance * 4;
            const glassesHeight = glassesWidth;
            const glassesX = mirroredX(eyeMidX + glassesWidth / 4.5);
            const glassesY = eyeMidY - glassesHeight * .6;

            return (
                <Image
                    image={glassesImage}
                    x={glassesX}
                    y={glassesY}
                    width={glassesWidth}
                    height={glassesHeight}
                    fit="contain"
                />
            );
        }

        // Mustache
        if (filter === 'mustache' && mustacheImage && nose) {
            const mustacheWidth = eyeDistance * 1.5;
            const mustacheHeight = mustacheWidth;
            const mustacheX = mirroredX(nose.x - mustacheWidth / 1.3);
            const mustacheY = nose.y - mustacheHeight/2;

            return (
                <Image
                    image={mustacheImage}
                    x={mustacheX - mustacheWidth/1.5}
                    y={mustacheY}
                    width={mustacheWidth}
                    height={mustacheHeight}
                    fit="contain"
                />
            );
        }

        return null;
    }, [landmarks, filter, hatImage, glassesImage, mustacheImage, viewWidth, shouldMirror]);

    const memoizedLandmarks = useMemo(() => {
        if (!landmarks) return null;

        const safeCircle = (
            point: { x: number; y: number } | undefined,
            r: number,
            color: string
        ) => {
            if (!point) return null;
            const x = mirrorX(point.x) + (viewWidth / 10 * 2.5); // w:360 h:750
            const y = point.y - (viewHeight / 15)
            return <Circle cx={x} cy={y} r={r} color={color} />;
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

    const cycleFilter = () => {
        const filters: FilterType[] = ['none', 'hat', 'glasses', 'mustache'];
        const nextIndex = (filters.indexOf(filter) + 1) % filters.length;
        setFilter(filters[nextIndex]);
    };

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
            <Canvas style={StyleSheet.absoluteFill}>
                {memoizedLandmarks}
                {memoizedFilter}
            </Canvas>

            <View style={styles.buttonContainer}>
                <TouchableOpacity style={styles.filterButton} onPress={cycleFilter}>
                    <Text style={styles.filterButtonText}>
                        {filter === 'none' ? 'Filter' : filter.toUpperCase()}
                    </Text>
                </TouchableOpacity>
            </View>
            {faceStatus && (
                <View style={styles.statusContainer}>
                    <Text style={styles.statusText}>Yaw: {faceStatus.yaw}</Text>
                    <Text style={styles.statusText}>Pitch: {faceStatus.pitch}</Text>
                    <Text style={styles.statusText}>Eyes: {faceStatus.eye}</Text>
                </View>
            )}
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
    buttonContainer: {
        position: 'absolute',
        bottom: 50,
        alignSelf: 'center',
    },
    filterButton: {
        backgroundColor: 'rgba(0,0,0,0.6)',
        paddingHorizontal: 20,
        paddingVertical: 12,
        borderRadius: 30,
        borderWidth: 2,
        borderColor: '#fff',
    },
    filterButtonText: {
        color: '#fff',
        fontWeight: 'bold',
        fontSize: 16,
    },
    statusContainer: {
        position: 'absolute',
        top: 50,
        left: 20,
        backgroundColor: 'rgba(0,0,0,0.5)',
        padding: 10,
        borderRadius: 10,
    },
    statusText: {
        color: 'lightgreen',
        fontSize: 14,
        fontWeight: 'bold',
    },
})