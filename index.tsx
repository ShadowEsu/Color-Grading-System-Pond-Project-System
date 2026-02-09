
import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";
import { 
  Upload, 
  MousePointer2, 
  CheckCircle2, 
  AlertCircle, 
  RefreshCcw, 
  Info,
  ChevronRight,
  Zap,
  Pipette,
  Crosshair,
  Camera,
  X
} from 'lucide-react';

// --- Color Science Utilities ---

interface RGB { r: number; g: number; b: number; }
interface LAB { l: number; a: number; b: number; }
interface ROI { x: number; y: number; w: number; h: number; }

class ColorScience {
  static rgbToHsv(r: number, g: number, b: number) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h = 0, s, v = max;
    const d = max - min;
    s = max === 0 ? 0 : d / max;
    if (max !== min) {
      switch (max) {
        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
        case g: h = (b - r) / d + 2; break;
        case b: h = (r - g) / d + 4; break;
      }
      h /= 6;
    }
    return { h, s, v };
  }

  static rgbToLab(r: number, g: number, b: number): LAB {
    let _r = r / 255, _g = g / 255, _b = b / 255;
    _r = _r > 0.04045 ? Math.pow((_r + 0.055) / 1.055, 2.4) : _r / 12.92;
    _g = _g > 0.04045 ? Math.pow((_g + 0.055) / 1.055, 2.4) : _g / 12.92;
    _b = _b > 0.04045 ? Math.pow((_b + 0.055) / 1.055, 2.4) : _b / 12.92;
    _r *= 100; _g *= 100; _b *= 100;
    const x = _r * 0.4124 + _g * 0.3576 + _b * 0.1805;
    const y = _r * 0.2126 + _g * 0.7152 + _b * 0.0722;
    const z = _r * 0.0193 + _g * 0.1192 + _b * 0.9505;
    const xn = 95.047, yn = 100.0, zn = 108.883;
    let fx = x / xn, fy = y / yn, fz = z / zn;
    fx = fx > 0.008856 ? Math.pow(fx, 1 / 3) : (7.787 * fx) + (16 / 116);
    fy = fy > 0.008856 ? Math.pow(fy, 1 / 3) : (7.787 * fy) + (16 / 116);
    fz = fz > 0.008856 ? Math.pow(fz, 1 / 3) : (7.787 * fz) + (16 / 116);
    return { l: (116 * fy) - 16, a: 500 * (fx - fy), b: 200 * (fy - fz) };
  }

  static deltaE(lab1: LAB, lab2: LAB): number {
    return Math.sqrt(Math.pow(lab1.l - lab2.l, 2) + Math.pow(lab1.a - lab2.a, 2) + Math.pow(lab1.b - lab2.b, 2));
  }

  static getMedianColor(ctx: CanvasRenderingContext2D, roi: ROI): { rgb: RGB, saturation: number } {
    const imageData = ctx.getImageData(roi.x, roi.y, roi.w, roi.h).data;
    const rs: number[] = [], gs: number[] = [], bs: number[] = [];
    const ss: number[] = [];
    for (let i = 0; i < imageData.length; i += 4) {
      const r = imageData[i], g = imageData[i + 1], b = imageData[i + 2];
      const { s, v } = this.rgbToHsv(r, g, b);
      if (v > 0.05 && v < 0.95) {
        rs.push(r); gs.push(g); bs.push(b); ss.push(s);
      }
    }
    const median = (arr: number[]) => {
      if (arr.length === 0) return 0;
      const sorted = [...arr].sort((a, b) => a - b);
      return sorted[Math.floor(sorted.length / 2)];
    };
    return {
      rgb: { r: median(rs), g: median(gs), b: median(bs) },
      saturation: ss.reduce((a, b) => a + b, 0) / (ss.length || 1)
    };
  }
}

// --- Main Application ---

const ROI_STEPS = [
  { key: 'A', label: 'Reference Color A', color: 'border-blue-500 bg-blue-500/10 text-blue-400' },
  { key: 'B', label: 'Reference Color B', color: 'border-emerald-500 bg-emerald-500/10 text-emerald-400' },
  { key: 'TEST', label: 'Unknown TEST Color', color: 'border-amber-500 bg-amber-500/10 text-amber-400' },
  { key: 'CONTROL', label: 'CONTROL Patch (White)', color: 'border-white bg-white/10 text-white' }
];

interface AnalysisResults {
  winner: string;
  pctA: number;
  pctB: number;
  dA: number;
  dB: number;
  report: string;
  controlSat: number;
}

export default function ColorExaminer() {
  const [image, setImage] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const [rois, setRois] = useState<Record<string, ROI | null>>({ A: null, B: null, TEST: null, CONTROL: null });
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentRect, setCurrentRect] = useState<ROI | null>(null);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      setImage(event.target?.result as string);
      setStep(0);
      setRois({ A: null, B: null, TEST: null, CONTROL: null });
      setResults(null);
      setError(null);
      setIsCameraActive(false);
    };
    reader.readAsDataURL(file);
  };

  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCameraActive(true);
      setImage(null);
    } catch (err: any) {
      setError(`Camera access failed: ${err.message}`);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
  };

  const capturePhoto = () => {
    if (videoRef.current) {
      const video = videoRef.current;
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = video.videoWidth;
      tempCanvas.height = video.videoHeight;
      const tCtx = tempCanvas.getContext('2d');
      if (tCtx) {
        tCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.95);
        setImage(dataUrl);
        setStep(0);
        setRois({ A: null, B: null, TEST: null, CONTROL: null });
        setResults(null);
        stopCamera();
      }
    }
  };

  useEffect(() => {
    if (image && canvasRef.current) {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current!;
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
        ctx.drawImage(img, 0, 0);
        updateDisplay();
      };
      img.src = image;
    }
  }, [image]);

  const updateDisplay = () => {
    if (!displayCanvasRef.current || !canvasRef.current) return;
    const dCanvas = displayCanvasRef.current;
    const sCanvas = canvasRef.current;
    const dCtx = dCanvas.getContext('2d')!;
    const container = containerRef.current;
    if (!container) return;
    const maxW = container.clientWidth;
    const maxH = 600;
    const ratio = Math.min(maxW / sCanvas.width, maxH / sCanvas.height);
    dCanvas.width = sCanvas.width * ratio;
    dCanvas.height = sCanvas.height * ratio;
    dCtx.drawImage(sCanvas, 0, 0, dCanvas.width, dCanvas.height);

    ROI_STEPS.forEach((stepItem) => {
      const roi = rois[stepItem.key];
      if (roi) {
        dCtx.strokeStyle = stepItem.key === 'CONTROL' ? '#fff' : stepItem.key === 'A' ? '#3b82f6' : stepItem.key === 'B' ? '#10b981' : '#f59e0b';
        dCtx.lineWidth = 3;
        dCtx.strokeRect(roi.x * ratio, roi.y * ratio, roi.w * ratio, roi.h * ratio);
        dCtx.fillStyle = dCtx.strokeStyle + '22';
        dCtx.fillRect(roi.x * ratio, roi.y * ratio, roi.w * ratio, roi.h * ratio);
      }
    });

    if (currentRect) {
      dCtx.strokeStyle = '#fff';
      dCtx.setLineDash([5, 5]);
      dCtx.strokeRect(currentRect.x * ratio, currentRect.y * ratio, currentRect.w * ratio, currentRect.h * ratio);
      dCtx.setLineDash([]);
    }
  };

  useEffect(() => { updateDisplay(); }, [rois, currentRect]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (step >= ROI_STEPS.length || !displayCanvasRef.current) return;
    const rect = displayCanvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setStartPos({ x, y });
    setIsDrawing(true);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing || !displayCanvasRef.current || !canvasRef.current) return;
    const rect = displayCanvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const sCanvas = canvasRef.current;
    const dCanvas = displayCanvasRef.current;
    const ratio = sCanvas.width / dCanvas.width;
    const roi = {
      x: Math.min(startPos.x, x) * ratio,
      y: Math.min(startPos.y, y) * ratio,
      w: Math.abs(x - startPos.x) * ratio,
      h: Math.abs(y - startPos.y) * ratio
    };
    setCurrentRect(roi);
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentRect) return;
    setIsDrawing(false);
    const key = ROI_STEPS[step].key;
    setRois(prev => ({ ...prev, [key]: currentRect }));
    setCurrentRect(null);
    setStep(prev => prev + 1);
  };

  const analyzeColors = async () => {
    if (!canvasRef.current) return;
    setIsAnalyzing(true);
    setError(null);
    try {
      const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true })!;
      const sampleA = ColorScience.getMedianColor(ctx, rois.A!);
      const sampleB = ColorScience.getMedianColor(ctx, rois.B!);
      const sampleTest = ColorScience.getMedianColor(ctx, rois.TEST!);
      const sampleControl = ColorScience.getMedianColor(ctx, rois.CONTROL!);
      const target = 240;
      const scaleR = target / Math.max(sampleControl.rgb.r, 1);
      const scaleG = target / Math.max(sampleControl.rgb.g, 1);
      const scaleB = target / Math.max(sampleControl.rgb.b, 1);
      const correct = (rgb: RGB) => ({
        r: Math.min(255, rgb.r * scaleR),
        g: Math.min(255, rgb.g * scaleG),
        b: Math.min(255, rgb.b * scaleB),
      });
      const cA = correct(sampleA.rgb);
      const cB = correct(sampleB.rgb);
      const cTest = correct(sampleTest.rgb);
      const labA = ColorScience.rgbToLab(cA.r, cA.g, cA.b);
      const labB = ColorScience.rgbToLab(cB.r, cB.g, cB.b);
      const labTest = ColorScience.rgbToLab(cTest.r, cTest.g, cTest.b);
      const dA = ColorScience.deltaE(labTest, labA);
      const dB = ColorScience.deltaE(labTest, labB);
      const totalD = dA + dB;
      const pctA = (dB / (totalD || 1)) * 100;
      const pctB = (dA / (totalD || 1)) * 100;

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const prompt = `Interpret chemical test result. Stats: WB Scaling: R=${scaleR.toFixed(2)}, G=${scaleG.toFixed(2)}, B=${scaleB.toFixed(2)}. Sat: ${(sampleControl.saturation * 100).toFixed(1)}%. Lab A: L=${labA.l.toFixed(1)}, a=${labA.a.toFixed(1)}, b=${labA.b.toFixed(1)}. Lab B: L=${labB.l.toFixed(1)}, a=${labB.a.toFixed(1)}, b=${labB.b.toFixed(1)}. Lab TEST: L=${labTest.l.toFixed(1)}, a=${labTest.a.toFixed(1)}, b=${labTest.b.toFixed(1)}. dE to A: ${dA.toFixed(2)}, dE to B: ${dB.toFixed(2)}. Winner %: ${(dA < dB ? pctA : pctB).toFixed(1)}%. Write professional analytical report (2-3 paras).`;

      const response = await ai.models.generateContent({ model: 'gemini-3-flash-preview', contents: prompt });
      const reportText = String(response.text || "Analysis complete but no descriptive report generated.");

      setResults({
        winner: dA < dB ? 'A' : 'B',
        pctA: Number(pctA),
        pctB: Number(pctB),
        dA: Number(dA),
        dB: Number(dB),
        report: reportText,
        controlSat: Number(sampleControl.saturation)
      });
    } catch (err: any) {
      console.error(err);
      setError(String(err.message || err || 'Analysis failed'));
    } finally { setIsAnalyzing(false); }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-4 md:p-8 font-sans selection:bg-blue-500/30">
      <header className="max-w-6xl mx-auto mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="bg-blue-600 p-2 rounded-lg shadow-lg shadow-blue-900/20"><Crosshair className="w-6 h-6 text-white" /></div>
            <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">Color Examiner</h1>
          </div>
          <p className="text-slate-500 font-medium">Professional Chemical Strip Vision Analysis</p>
        </div>
        
        <div className="flex items-center gap-2">
          {!image && !isCameraActive ? (
            <>
              <button 
                onClick={startCamera} 
                className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-white px-6 py-3 rounded-full font-bold transition-all"
              >
                <Camera className="w-5 h-5" />
                Live Camera
              </button>
              <label className="group flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-full font-bold cursor-pointer transition-all shadow-lg shadow-blue-600/20">
                <Upload className="w-5 h-5 group-hover:-translate-y-1 transition-transform" />
                Upload Image
                <input type="file" className="hidden" accept="image/*" onChange={handleUpload} />
              </label>
            </>
          ) : (
            <button 
              onClick={() => { 
                setImage(null); 
                setResults(null); 
                setStep(0); 
                setRois({ A: null, B: null, TEST: null, CONTROL: null }); 
                stopCamera();
              }} 
              className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-300 px-6 py-3 rounded-full font-bold transition-all"
            >
              <RefreshCcw className="w-5 h-5" />
              Reset All
            </button>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-7 space-y-6">
          {!image && !isCameraActive ? (
            <div className="aspect-video rounded-2xl border-2 border-dashed border-slate-800 flex flex-col items-center justify-center p-12 bg-slate-900/50 group transition-all hover:border-blue-500/50">
              <div className="bg-slate-800 p-6 rounded-full mb-4 group-hover:scale-110 transition-transform"><Pipette className="w-12 h-12 text-slate-600" /></div>
              <h3 className="text-xl font-semibold mb-2 text-slate-300">No Image Selected</h3>
              <p className="text-slate-500 text-center max-w-sm mb-6">Select a method above to begin. You can either take a live photo or upload a saved one.</p>
              <div className="flex gap-4">
                <button onClick={startCamera} className="bg-slate-800 hover:bg-slate-700 text-white px-6 py-2 rounded-lg transition-all flex items-center gap-2">
                  <Camera className="w-4 h-4" /> Camera
                </button>
                <label className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg cursor-pointer transition-all flex items-center gap-2">
                  <Upload className="w-4 h-4" /> Upload
                  <input type="file" className="hidden" accept="image/*" onChange={handleUpload} />
                </label>
              </div>
            </div>
          ) : isCameraActive ? (
            <div className="bg-black rounded-3xl border border-slate-800 overflow-hidden shadow-2xl relative aspect-video flex items-center justify-center">
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                className="w-full h-full object-cover"
              />
              <div className="absolute bottom-6 left-0 right-0 flex justify-center gap-4">
                <button 
                  onClick={stopCamera} 
                  className="bg-slate-900/80 hover:bg-slate-800 text-white p-4 rounded-full backdrop-blur-md transition-all border border-white/10"
                >
                  <X className="w-6 h-6" />
                </button>
                <button 
                  onClick={capturePhoto} 
                  className="bg-blue-600 hover:bg-blue-500 text-white px-8 py-4 rounded-full font-bold shadow-2xl flex items-center gap-2 active:scale-95 transition-all"
                >
                  <Camera className="w-6 h-6" />
                  Capture Photo
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-slate-900 rounded-3xl border border-slate-800 overflow-hidden shadow-2xl">
              <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between">
                <div className="flex items-center gap-2"><MousePointer2 className="w-4 h-4 text-blue-400" /><span className="text-sm font-semibold text-slate-400">ROI SELECTION MODE</span></div>
                <div className="flex gap-2">{ROI_STEPS.map((s, idx) => (<div key={String(s.key)} className={`w-2 h-2 rounded-full ${idx < step ? 'bg-emerald-500' : idx === step ? 'bg-blue-500 animate-pulse' : 'bg-slate-700'}`} />))}</div>
              </div>
              <div ref={containerRef} className="relative cursor-crosshair bg-black flex items-center justify-center overflow-hidden min-h-[400px]">
                <canvas ref={displayCanvasRef} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} className="max-w-full h-auto" />
                <canvas ref={canvasRef} className="hidden" />
              </div>
              <div className="p-4 bg-slate-900/80 border-t border-slate-800 flex justify-between items-center">
                <div className="text-sm">
                  {step < ROI_STEPS.length ? (
                    <div className="flex items-center gap-3"><div className={`px-3 py-1 rounded-md border ${String(ROI_STEPS[step].color)}`}>Select {String(ROI_STEPS[step].label)}</div><span className="text-slate-500 italic">Drag a rectangle over the target area</span></div>
                  ) : (<div className="flex items-center gap-2 text-emerald-400 font-medium"><CheckCircle2 className="w-5 h-5" />All regions selected</div>)}
                </div>
                {step === ROI_STEPS.length && !results && (<button onClick={analyzeColors} disabled={isAnalyzing} className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white px-8 py-2 rounded-xl font-bold transition-all flex items-center gap-2 shadow-lg shadow-blue-600/20">{isAnalyzing ? <Zap className="w-5 h-5 animate-spin" /> : <ChevronRight className="w-5 h-5" />}{isAnalyzing ? 'Analyzing...' : 'Execute Analysis'}</button>)}
              </div>
            </div>
          )}
        </div>
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-slate-900 rounded-3xl border border-slate-800 p-6">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2"><Pipette className="w-5 h-5 text-blue-400" />Capture Workflow</h3>
            <div className="space-y-3">
              {ROI_STEPS.map((s, idx) => (
                <div key={String(s.key)} className={`flex items-center justify-between p-3 rounded-2xl border transition-all ${rois[s.key] ? 'bg-slate-800/40 border-slate-700' : step === idx ? 'bg-blue-500/5 border-blue-500/30' : 'border-transparent opacity-40'}`}>
                  <div className="flex items-center gap-3"><div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${rois[s.key] ? 'bg-emerald-500 text-emerald-950' : step === idx ? 'bg-blue-500 text-blue-950' : 'bg-slate-800 text-slate-500'}`}>{idx + 1}</div><span className={`font-semibold ${step === idx ? 'text-white' : 'text-slate-400'}`}>{String(s.label)}</span></div>
                  {rois[s.key] && <CheckCircle2 className="w-5 h-5 text-emerald-500" />}
                </div>
              ))}
            </div>
          </div>
          {results ? (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-700 space-y-6">
              <div className="bg-slate-900 rounded-3xl border border-slate-800 overflow-hidden">
                <div className="p-6 bg-gradient-to-br from-slate-900 to-slate-950">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold text-white">Analysis Result</h3>
                    <div className={`px-4 py-1 rounded-full text-xs font-bold uppercase tracking-widest ${results.winner === 'A' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'}`}>Closer to {String(results.winner)}</div>
                  </div>
                  <div className="space-y-6 mb-8">
                    <div>
                      <div className="flex justify-between text-sm mb-2"><span className="text-slate-400 font-medium">Likelihood: Color A</span><span className="text-white font-bold">{String(results.pctA.toFixed(1))}%</span></div>
                      <div className="h-3 bg-slate-800 rounded-full overflow-hidden"><div className="h-full bg-blue-500 transition-all duration-1000" style={{ width: `${results.pctA}%` }} /></div>
                      <div className="flex justify-between text-[10px] mt-1 text-slate-600 uppercase tracking-tighter font-bold"><span>DeltaE: {String(results.dA.toFixed(2))}</span></div>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-2"><span className="text-slate-400 font-medium">Likelihood: Color B</span><span className="text-white font-bold">{String(results.pctB.toFixed(1))}%</span></div>
                      <div className="h-3 bg-slate-800 rounded-full overflow-hidden"><div className="h-full bg-emerald-500 transition-all duration-1000" style={{ width: `${results.pctB}%` }} /></div>
                      <div className="flex justify-between text-[10px] mt-1 text-slate-600 uppercase tracking-tighter font-bold"><span>DeltaE: {String(results.dB.toFixed(2))}</span></div>
                    </div>
                  </div>
                  {results.controlSat > 0.15 && (
                    <div className="flex items-start gap-3 p-3 bg-amber-500/10 border border-amber-500/30 rounded-2xl mb-6">
                      <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                      <div><p className="text-sm font-bold text-amber-500">Calibration Warning</p><p className="text-xs text-amber-200/70 leading-relaxed">Control patch saturation is high ({String((results.controlSat * 100).toFixed(0))}%). Results may be biased by ambient lighting.</p></div>
                    </div>
                  )}
                  <div className="bg-slate-800/30 border border-slate-700/50 p-5 rounded-2xl relative">
                    <div className="absolute -top-3 left-4 bg-slate-700 px-3 py-1 rounded-md text-[10px] font-bold text-white uppercase tracking-widest flex items-center gap-2"><Info className="w-3 h-3" />Technical Audit</div>
                    <div className="prose prose-invert prose-sm mt-2 max-h-[300px] overflow-y-auto custom-scrollbar">
                      {results.report.split('\n').map((para, i) => (<p key={i} className="text-slate-400 leading-relaxed mb-3 last:mb-0 italic font-medium">{String(para)}</p>))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-slate-900/50 rounded-3xl border border-slate-800 p-8 flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mb-6"><Zap className="w-8 h-8 text-slate-600" /></div>
              <h4 className="text-slate-400 font-bold mb-2">Analysis Pending</h4>
              <p className="text-slate-600 text-sm max-w-[200px]">Complete all 4 selections to unlock the perceptual analysis report.</p>
            </div>
          )}
        </div>
      </main>
      {error && (
        <div className="fixed bottom-8 right-8 bg-red-500/10 border border-red-500/30 text-red-400 p-4 rounded-2xl flex items-center gap-3 backdrop-blur-xl shadow-2xl animate-in slide-in-from-right-8 max-w-sm z-50">
          <AlertCircle className="w-6 h-6 flex-shrink-0" />
          <span className="font-bold text-sm">{String(error)}</span>
          <button onClick={() => setError(null)} className="ml-2 text-white/50 hover:text-white">&times;</button>
        </div>
      )}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #475569; }
      `}</style>
    </div>
  );
}

const root = createRoot(document.getElementById('root')!);
root.render(<ColorExaminer />);
