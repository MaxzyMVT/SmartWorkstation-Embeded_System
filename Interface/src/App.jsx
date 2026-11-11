import React, { useState, useEffect } from 'react';
import { RotateCcw, CheckCircle2, XCircle, Edit3, Save, Armchair, User, Sun, Thermometer, Volume2 } from 'lucide-react';

const DEFAULT_THRESHOLDS = {
  distance: { min: 30, max: 60, unit: 'cm' },
  light: { min: 300, max: 990, unit: 'Lux' },
  temp: { min: 22, max: 27, unit: '°C' },
  sound: { min: 0, max: 40, unit: 'dB' }
};

export default function App() {
  const [isEditing, setIsEditing] = useState(false);
  const [thresholds, setThresholds] = useState(DEFAULT_THRESHOLDS);
  
  const [sensorValues, setSensorValues] = useState({
    seat: true,
    distance: 50,
    light: 430,
    temp: 24.7,
    sound: 35
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setSensorValues(prev => ({
        ...prev,
        distance: prev.distance + (Math.random() > 0.5 ? 1 : -1),
        light: prev.light + Math.floor(Math.random() * 10 - 5),
        temp: +(prev.temp + (Math.random() * 0.2 - 0.1)).toFixed(1),
        sound: prev.sound + Math.floor(Math.random() * 4 - 2)
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const isGlobalError = 
    !sensorValues.seat ||
    sensorValues.distance < thresholds.distance.min || sensorValues.distance > thresholds.distance.max ||
    sensorValues.light < thresholds.light.min || sensorValues.light > thresholds.light.max ||
    sensorValues.temp < thresholds.temp.min || sensorValues.temp > thresholds.temp.max ||
    sensorValues.sound < thresholds.sound.min || sensorValues.sound > thresholds.sound.max;

  const handleThresholdChange = (sensor, type, value) => {
    setThresholds(prev => ({
      ...prev,
      [sensor]: { ...prev[sensor], [type]: Number(value) }
    }));
  };

  const resetToDefault = () => {
    setThresholds(DEFAULT_THRESHOLDS);
    setIsEditing(false);
  };

  const toggleSeat = () => {
    setSensorValues(prev => ({ ...prev, seat: !prev.seat }));
  };

  const isSeatError = !sensorValues.seat;

  return (
    <div className="min-h-screen bg-[#F4F7F9] p-8 font-sans text-slate-800 flex flex-col gap-4">
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 flex flex-col gap-4">
        <div className="flex items-center gap-4">
          <h1 className="text-4xl font-bold tracking-tight">Smart Workstation</h1>
          <span className="bg-slate-100 text-slate-500 px-3 py-1 rounded-md text-sm font-medium">Auto-Detect</span>
        </div>
        <div className="flex items-center gap-3">
          {isEditing ? (
            <button onClick={() => setIsEditing(false)} className="flex items-center gap-2 bg-[#22C55E] text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-green-600 transition-colors"><Save size={16} /> Save</button>
          ) : (
            <button onClick={() => setIsEditing(true)} className="flex items-center gap-2 bg-white border border-slate-200 text-slate-600 px-4 py-2 rounded-lg text-sm font-medium hover:bg-slate-50 transition-colors"><Edit3 size={16} /> Edit</button>
          )}
          <button onClick={resetToDefault} className="flex items-center gap-2 bg-white border border-slate-200 text-slate-600 px-4 py-2 rounded-lg text-sm font-medium hover:bg-slate-50 transition-colors"><RotateCcw size={16} /> Reset to default</button>
        </div>
      </div>

      <div className={`rounded-xl p-4 flex items-center gap-4 border transition-colors ${isGlobalError ? 'bg-[#FFF0F2] border-[#FFCBD3]' : 'bg-[#E5FDEB] border-green-200'}`}>
        {isGlobalError ? <XCircle className="text-red-500 w-8 h-8 flex-shrink-0" /> : <CheckCircle2 className="text-green-600 w-8 h-8 flex-shrink-0" />}
        <div>
          <h2 className="text-xl font-bold mb-0.5">{isGlobalError ? 'Curtain is close' : 'Curtain is open'}</h2>
          <p className="text-sm opacity-80">{isGlobalError ? 'Unsuitable Environment' : 'Suitable Environment'}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        {/* Seat Card */}
        <div className={`aspect-square rounded-2xl p-6 border flex flex-col transition-colors ${isSeatError ? 'bg-[#FFF0F2] border-[#FFCBD3]' : 'bg-white border-slate-100'}`}>
          <div className="flex items-center gap-3 font-bold text-xl">
            <div className={`p-2 rounded-lg border ${isSeatError ? 'bg-[#FFF0F2] text-red-500 border-[#FFCBD3]' : 'bg-[#E5FDEB] text-[#22C55E] border-green-200'}`}>
              <Armchair size={24} />
            </div>
            Seat
          </div>
          <div className="flex-1 flex items-center justify-center text-6xl font-bold tracking-tight">
            {sensorValues.seat ? 'YES' : 'No'}
          </div>
          {/* Spacer block to perfectly align the centered value with other cards */}
          <div className="h-[76px]"></div>
        </div>

        {/* Other Sensors */}
        {[
          { key: 'distance', label: 'Distance', val: sensorValues.distance, Icon: User },
          { key: 'light', label: 'Light', val: sensorValues.light, Icon: Sun },
          { key: 'temp', label: 'Temperature', val: sensorValues.temp, Icon: Thermometer },
          { key: 'sound', label: 'Sound', val: sensorValues.sound, Icon: Volume2 }
        ].map((sensor) => {
          const isCardError = sensorValues[sensor.key] < thresholds[sensor.key].min || sensorValues[sensor.key] > thresholds[sensor.key].max;
          const Icon = sensor.Icon;
          
          return (
            <div key={sensor.key} className={`aspect-square rounded-2xl p-6 border flex flex-col transition-colors ${isCardError ? 'bg-[#FFF0F2] border-[#FFCBD3]' : 'bg-white border-slate-100'}`}>
              <div className="flex items-center gap-3 font-bold text-xl">
                <div className={`p-2 rounded-lg border ${isCardError ? 'bg-[#FFF0F2] text-red-500 border-[#FFCBD3]' : 'bg-[#E5FDEB] text-[#22C55E] border-green-200'}`}>
                  <Icon size={24} />
                </div>
                {sensor.label}
              </div>
              
              <div className="flex-1 flex items-center justify-center text-6xl font-bold tracking-tight gap-1">
                {sensor.val}
                <span className="text-sm font-medium text-slate-400 mt-6">{thresholds[sensor.key].unit}</span>
              </div>

              <div className="h-[76px] rounded-xl p-3 bg-[#F4F7F9] flex flex-col justify-center">
                {isEditing ? (
                  <div className="flex items-center gap-3 justify-center w-full">
                    <div className="flex flex-col items-start gap-1 w-1/2">
                      <span className="text-[10px] font-bold text-slate-400 capitalize">Min</span>
                      <input 
                        type="number" 
                        value={thresholds[sensor.key].min} 
                        onChange={(e) => handleThresholdChange(sensor.key, 'min', e.target.value)} 
                        className="w-full px-2 py-1.5 text-sm border border-slate-200 rounded text-left font-medium text-slate-700 bg-white focus:outline-none focus:border-slate-300"
                      />
                    </div>
                    <div className="flex flex-col items-start gap-1 w-1/2">
                      <span className="text-[10px] font-bold text-slate-400 capitalize">Max</span>
                      <input 
                        type="number" 
                        value={thresholds[sensor.key].max} 
                        onChange={(e) => handleThresholdChange(sensor.key, 'max', e.target.value)} 
                        className="w-full px-2 py-1.5 text-sm border border-slate-200 rounded text-left font-medium text-slate-700 bg-white focus:outline-none focus:border-slate-300"
                      />
                    </div>
                  </div>
                ) : (
                  <div className="text-center">
                    <div className="text-[10px] font-bold text-slate-400 mb-0.5">ACCEPTABLE RANGE</div>
                    <div className="text-sm font-medium text-slate-500">
                      {thresholds[sensor.key].min} - {thresholds[sensor.key].max} {thresholds[sensor.key].unit}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}