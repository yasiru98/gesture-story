import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import DebugPose from './pages/DebugPose';

function App() {
  const [count, setCount] = useState(0)

  return <DebugPose />; 
}

export default App
