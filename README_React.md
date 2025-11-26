# Text2VR React Frontend

A React-based web interface for Text2VR.  
Users describe a scene in natural language and the system generates a 360° panorama that can be experienced in VR.

## Key Features

### 🎯 Core Features
- **Natural language input**: Generate panorama scenes from simple text descriptions
- **Real-time status updates**: Monitor the generation process live
- **VR viewer**: Immersive 360° panorama experience using A-Frame
- **Responsive design**: Optimized for both mobile and desktop

### 🌟 VR Features
- Look around with mouse or touch
- VR headset support
- High-quality 360° panorama rendering
- 2D preview available alongside VR view

## Tech Stack

- **Frontend**: React 19 + TypeScript
- **3D/VR**: A-Frame
- **Build Tool**: Vite
- **Styling**: Inline CSS with gradient effects

## Development & Run

### Development mode
```bash
npm run dev
````

### Build

```bash
npm run build
```

### Production server

The Python FastAPI backend serves the built React app:

```bash
python main.py
```

## Project Structure

```txt
src/
├── components/
│   ├── InputForm.tsx          # Text prompt input form
│   ├── StatusSection.tsx      # Generation status display
│   ├── LoadingSpinner.tsx     # Loading spinner
│   └── VRPanoramaViewer.tsx   # VR panorama viewer
├── services/
│   └── apiService.ts          # API communication service
├── types/
│   ├── api.ts                 # API type definitions
│   └── aframe.d.ts            # A-Frame type definitions
├── App.tsx                    # Main app component
├── App.css                    # Global styles
└── main.tsx                   # React entry point
```

## API Integration

Communicates with the FastAPI backend using the following endpoints:

* `POST /generate` – Start panorama generation
* `GET /status/{task_id}` – Check generation status
* `GET /panorama/{task_id}` – Fetch the final panorama image
* `GET /health` – Health check for the server

## How to Use

1. **Enter scene description**: e.g., “Sunlight streaming through trees in a peaceful forest”
2. **Enter scene name (optional)**: Use a custom name or let the system auto-generate one
3. **Start generation**: Click the “Generate Panorama” button
4. **Track progress**: Monitor the generation process in real time
5. **Experience in VR**: View the final panorama in the VR viewer

## VR Controls

* **Mouse**: Click and drag to rotate the view
* **Touch**: Touch and drag to rotate the view
* **VR headset**: Click the VR button to enter immersive mode

## Development Notes

* Uses the latest features of React 19
* Ensures type safety with TypeScript
* Generates A-Frame elements dynamically to keep TypeScript compatibility
* Uses Vite proxy configuration for API integration during development
* Responsive CSS for a consistent experience across devices
