import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

const socket = io('http://localhost:5000');

const VideoFeed = () => {
    const [imageSrc, setImageSrc] = useState('');

    useEffect(() => {
        socket.on('frame', (data) => {
            setImageSrc(`data:image/jpeg;base64,${data.frame}`);
        });

        return () => {
            socket.off('frame');
        };
    }, []);

    const handleStartRecording = () => {
        console.log('Start recording button clicked');
        socket.emit('start_recording');
    };

    return (
        <div>
            <img src={imageSrc} alt="Video Feed" style={{ width: '100%' }} />
            <button onClick={handleStartRecording} style={{ position: 'fixed', bottom: '20px', left: '50%', transform: 'translateX(-50%)' }}>
                Start Recording
            </button>
        </div>
    );
};

export default VideoFeed;
