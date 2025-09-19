import { useState } from 'react'
import './App.css'
import DebugPose from './pages/DebugPose';

function App() {
  const [count, setCount] = useState(0)

  return <DebugPose />; 
}

export default App
