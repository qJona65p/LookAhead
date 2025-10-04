import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet, Text, useWindowDimensions } from 'react-native';
import { Camera as VisionCamera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { Camera, Face, FaceDetectionOptions, Landmarks } from 'react-native-vision-camera-face-detector';
import { Canvas, Rect } from '@shopify/react-native-skia';
import Animated, { useSharedValue, useAnimatedStyle, withTiming } from 'react-native-reanimated';

export default function CameraScreen() {
    const { width, height } = useWindowDimensions();
    const { hasPermission, requestPermission } = useCameraPermission();
    const [ faceStatus, setFaceStatus ] = useState<{ yaw: string; pitch: string; eye: string } | null>(null);
    const device = useCameraDevice('front');

    // Permission
    useEffect(() => {
        if (!hasPermission) {
            requestPermission();
        }
    }, [hasPermission, requestPermission]);

    if (hasPermission === null) {
        return (<View style={styles.container}><Text>Rquesting camera permission</Text></View>);
        
    }
    if (hasPermission === false) {
        return (<View style={styles.container}><Text>No camera access</Text></View>);
    }

    if (!device) {
        return (<View style={styles.container}><Text>No front camera available</Text></View>);
    }

    const aFaceW = useSharedValue(0);
    const aFaceH = useSharedValue(0);
    const aFaceX = useSharedValue(0);
    const aFaceY = useSharedValue(0);
    const lMouthLX = useSharedValue(0);
    const lMouthLY = useSharedValue(0);

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
    const drawFaceLandmarks = (landmarks?: Landmarks) => {
        if (landmarks) {
            const { x, y } = landmarks.MOUTH_LEFT;
        } else {
            aFaceW.value = aFaceH.value = aFaceX.value = aFaceY.value = 0;
        }
    }

    const faceBoxStyle = useAnimatedStyle(() => ({
        position: 'absolute',
        borderWidth: 4,
        borderLeftColor: 'rgb(0,255,0)',
        borderRightColor: 'rgb(0,255,0)',
        borderBottomColor: 'rgb(0,255,0)',
        borderTopColor: 'rgb(0,255,0)',
        width: withTiming(aFaceW.value, { duration: 100 }),
        height: withTiming(aFaceH.value, { duration: 100 }),
        left: withTiming(aFaceX.value, { duration: 100 }),
        top: withTiming(aFaceY.value, { duration: 100 })
    }));

    const faceDetectionOptions = useRef<FaceDetectionOptions>({
        performanceMode: 'fast',
        landmarkMode: 'all',
        contourMode: 'none',
        classificationMode: 'all',
        trackingEnabled: false,
        windowWidth: width,
        windowHeight: height,
        autoMode: true,
    }).current;

    const handleFacesDetection = (faces: Face[]) => {
        try {
            if (faces?.length > 0) {
                const face = faces[0];

                // You can add your own logic here!!
                drawFaceBounds(face);
                drawFaceLandmarks(face.landmarks);
                setFaceStatus({
                    yaw: face.yawAngle > 15 ? "Right" : face.yawAngle < -15 ? "Left" : "Center",
                    pitch: face.pitchAngle > 15 ? "Up" : face.pitchAngle < -10 ? "Down" : "Center",
                    eye: face.leftEyeOpenProbability > 0.7 && face.rightEyeOpenProbability > 0.7 ? "Open" : "Close"
                });
            } else {
                drawFaceBounds();
            }
        } catch (error) {
            console.error("Error in face detection:", error);
        }
    }

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