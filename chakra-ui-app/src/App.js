import React from 'react';
import './App.css';
import VideoFeed from './VideoFeed';

function App() {
    return (
        <div className="App">
            <div style={{ display: 'flex', flexWrap: 'wrap', height: '100vh' }}>
                <div style={{ flex: '1 0 50%', border: '1px solid black' }}>
                    <VideoFeed />
                </div>
                <div style={{ flex: '1 0 50%', border: '1px solid black' }}>
                    {/* Another component or content */}
                </div>
                <div style={{ flex: '1 0 50%', border: '1px solid black' }}>
                    {/* Another component or content */}
                </div>
                <div style={{ flex: '1 0 50%', border: '1px solid black' }}>
                    {/* Another component or content */}
                </div>
            </div>
        </div>
    );
}

export default App;
