import React, { useState, useRef, useEffect } from 'react';
import { View, StyleSheet, Text, Dimensions } from 'react-native';
import { Camera, CameraView } from 'expo-camera';
import * as Reanimated from 'react-native-reanimated';
import { Canvas, Rect } from '@shopify/react-native-skia';
import { useFacesInPhoto } from '@infinitered/react-native-mlkit-face-detection';

const { width: screenWidth, height: screenHeight } = Dimensions.get("window")

export default function CameraScreen() {
    const [hasPermission, setHasPermission] = useState<boolean | null>(null);
    const [facing, setFacing] = useState<'front' | 'back'>('front');
    const [imageUri, setImageUri] = useState<string | null>(null);
    const cameraRef = useRef<CameraView>(null);
    const { faces, error, status } = useFacesInPhoto(imageUri || '');

    // Permission
    useEffect(() => {
        (async () => {
            const { status: permStatus } = await Camera.requestCameraPermissionsAsync();
            setHasPermission(permStatus === "granted");
        })();
    }, []);

    // Capture frame for ML Kit processing
    const captureAndDetect = async () => {
        if (cameraRef.current) {
            try {
                const photo = await cameraRef.current.takePictureAsync({ skipProcessing: true, base64: false });
                setImageUri(photo.uri);
            } catch (err) {
                console.error('Capture error:', err);
            }
        }
    };

    // Real-time frame capture
    useEffect(() => {
        const interval = setInterval(captureAndDetect, 150);
        return () => clearInterval(interval);
    }, []);

    if (hasPermission === null) {
        return (
            <View style={styles.container}>
                <Text>Rquesting camera permission</Text>
            </View>
        );

    }
    if (hasPermission === false) {
        return (
            <View style={styles.container}>
                <Text>No camera access</Text>
            </View>
        );
    }

    if (error) {
        return (
            <View style={styles.container}>
                <Text>Detection error: {error}</Text>
            </View>
        );
    }

    if (status === 'modelLoading') {
        return (
            <View style={styles.container}>
                <Text>Loading ML model</Text>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <CameraView
                style={styles.camera}
                ref={cameraRef}
                facing={facing}
                onMountError={(error) => console.log('Camera error:', error.message)}
            />
            {faces?.length === 0 && <Text style={styles.noFace}>No face detected</Text>}
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
    noFace: {
        position: 'absolute',
        bottom: 50,
        alignSelf: 'center',
        color: 'white',
        fontSize: 18
    },
    blush: {
        position: 'absolute',
        width: 30,
        height: 30
    },
    blushCanvas: {
        flex: 1
    }
})
