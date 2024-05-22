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

    return (
        <div>
            <img src={imageSrc} alt="Video Feed" style={{ width: '100%' }} />
        </div>
    );
};

export default VideoFeed;
