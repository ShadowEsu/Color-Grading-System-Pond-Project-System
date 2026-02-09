
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
  X,
  Share2,
  Download,
  ShieldCheck,
  Microscope,
  FileText,
  Copy,
  ExternalLink
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

// --- Constants ---

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

// --- Main Application ---

export default function ColorExaminer() {
  const [image, setImage] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const [rois, setRois] = useState<Record<string, ROI | null>>({ A: null, B: null, TEST: null, CONTROL: null });
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);

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
      resetAnalysis();
    };
    reader.readAsDataURL(file);
  };

  const resetAnalysis = () => {
    setStep(0);
    setRois({ A: null, B: null, TEST: null, CONTROL: null });
    setResults(null);
    setError(null);
    setIsCameraActive(false);
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
        resetAnalysis();
        stopCamera();
      }
    }
  };

  const copyResultsToClipboard = () => {
    if (!results) return;
    const text = `Color Examiner AI Results:\nWinner: ${results.winner}\nLikelihood A: ${results.pctA.toFixed(1)}%\nLikelihood B: ${results.pctB.toFixed(1)}%\n\nTechnical Report:\n${results.report}`;
    navigator.clipboard.writeText(text);
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2000);
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
    <div className="min-h-screen flex flex-col">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-xl border-b border-slate-900 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-xl shadow-lg shadow-blue-900/40">
              <Crosshair className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-extrabold tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                Color Examiner AI
              </h1>
              <span className="text-[10px] uppercase tracking-widest font-bold text-blue-500">Professional Studio</span>
            </div>
          </div>

          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-400">
            <a href="#" className="hover:text-white transition-colors">Analyzer</a>
            <a href="#features" className="hover:text-white transition-colors">How it works</a>
            <a href="#about" className="hover:text-white transition-colors">Science</a>
          </div>

          <div className="flex items-center gap-3">
            {image || isCameraActive ? (
              <button 
                onClick={() => { setImage(null); setResults(null); stopCamera(); resetAnalysis(); }}
                className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-full text-sm font-bold transition-all"
              >
                <RefreshCcw className="w-4 h-4" />
                Reset
              </button>
            ) : null}
            <button className="bg-white text-slate-950 px-5 py-2 rounded-full text-sm font-bold hover:bg-slate-200 transition-all flex items-center gap-2">
              <Share2 className="w-4 h-4" /> Share
            </button>
          </div>
        </div>
      </nav>

      <main className="flex-grow">
        {!image && !isCameraActive ? (
          /* Hero Section */
          <div className="relative overflow-hidden pt-20 pb-32">
            <div className="max-w-7xl mx-auto px-6 relative z-10 text-center">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-bold mb-8 animate-in fade-in slide-in-from-bottom-2 duration-700">
                <Zap className="w-3 h-3" /> Powered by Gemini 3 & CIE Lab
              </div>
              <h2 className="text-5xl md:text-7xl font-black mb-6 leading-tight tracking-tight animate-in fade-in slide-in-from-bottom-4 duration-700 delay-100">
                Precision Chemical <br className="hidden md:block" />
                Vision for the <span className="text-blue-500 italic">Modern Lab</span>.
              </h2>
              <p className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto mb-12 animate-in fade-in slide-in-from-bottom-6 duration-700 delay-200">
                Accurately compare test strip results using perceptual color space math. 
                Our AI corrects for lighting, glare, and shadows in real-time.
              </p>

              <div className="flex flex-col sm:flex-row items-center justify-center gap-6 animate-in fade-in slide-in-from-bottom-8 duration-700 delay-300">
                <button 
                  onClick={startCamera}
                  className="w-full sm:w-auto flex items-center justify-center gap-3 bg-blue-600 hover:bg-blue-500 text-white px-10 py-5 rounded-2xl font-black text-lg transition-all shadow-2xl shadow-blue-600/30 group active:scale-95"
                >
                  <Camera className="w-6 h-6 group-hover:scale-110 transition-transform" />
                  Live Camera Capture
                </button>
                <label className="w-full sm:w-auto flex items-center justify-center gap-3 bg-slate-900 hover:bg-slate-800 border border-slate-800 text-white px-10 py-5 rounded-2xl font-black text-lg cursor-pointer transition-all active:scale-95">
                  <Upload className="w-6 h-6" />
                  Upload Photo
                  <input type="file" className="hidden" accept="image/*" onChange={handleUpload} />
                </label>
              </div>

              {/* Feature Grid */}
              <div id="features" className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-32 text-left">
                {[
                  { icon: ShieldCheck, title: "Lighting Neutral", desc: "Uses white-patch calibration to scale RGB channels and remove color casts." },
                  { icon: Microscope, title: "Perceptual Lab Space", desc: "Calculates differences using CIE Lab DeltaE distances, not raw RGB pixels." },
                  { icon: FileText, title: "AI Generated Reports", desc: "Detailed technical explanation for every test result, powered by Gemini." }
                ].map((f, i) => (
                  <div key={i} className="bg-slate-900/50 border border-slate-800 p-8 rounded-3xl hover:border-slate-700 transition-colors">
                    <f.icon className="w-10 h-10 text-blue-500 mb-6" />
                    <h4 className="text-xl font-bold mb-3">{f.title}</h4>
                    <p className="text-slate-500 leading-relaxed text-sm">{f.desc}</p>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Background Glows */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-blue-600/5 blur-[120px] pointer-events-none rounded-full" />
            <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-emerald-500/5 blur-[100px] pointer-events-none rounded-full" />
          </div>
        ) : (
          /* Editor Section */
          <div className="max-w-7xl mx-auto px-6 py-12">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
              {/* Left Column: Visual Editor */}
              <div className="lg:col-span-8 space-y-8">
                {isCameraActive ? (
                  <div className="bg-black rounded-[40px] border border-slate-800 overflow-hidden shadow-2xl relative aspect-video flex items-center justify-center group">
                    <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <div className="absolute bottom-10 left-0 right-0 flex justify-center gap-6">
                      <button 
                        onClick={stopCamera} 
                        className="bg-slate-900/80 hover:bg-slate-800 text-white p-5 rounded-full backdrop-blur-xl transition-all border border-white/10 active:scale-90"
                      >
                        <X className="w-6 h-6" />
                      </button>
                      <button 
                        onClick={capturePhoto} 
                        className="bg-white hover:bg-slate-100 text-slate-950 px-10 py-5 rounded-full font-black shadow-2xl flex items-center gap-3 active:scale-95 transition-all"
                      >
                        <Camera className="w-6 h-6" /> Capture Test
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="bg-slate-900 rounded-[40px] border border-slate-800 overflow-hidden shadow-2xl flex flex-col">
                    <div className="px-8 py-6 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className="bg-blue-500/10 p-2 rounded-lg">
                          <MousePointer2 className="w-5 h-5 text-blue-400" />
                        </div>
                        <div>
                          <h3 className="font-bold text-lg text-white">Region Selection</h3>
                          <p className="text-xs text-slate-500 uppercase tracking-widest font-bold">Step {step + 1} of 4</p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {ROI_STEPS.map((s, idx) => (
                          <div key={String(s.key)} className={`w-10 h-1.5 rounded-full transition-all duration-500 ${idx < step ? 'bg-emerald-500' : idx === step ? 'bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.5)]' : 'bg-slate-800'}`} />
                        ))}
                      </div>
                    </div>
                    
                    <div ref={containerRef} className="relative cursor-crosshair bg-black flex items-center justify-center overflow-hidden min-h-[500px]">
                      <canvas ref={displayCanvasRef} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} className="max-w-full h-auto" />
                      <canvas ref={canvasRef} className="hidden" />
                      
                      {step < ROI_STEPS.length && (
                        <div className="absolute top-6 left-6 pointer-events-none animate-in slide-in-from-left-4 duration-500">
                          <div className={`px-6 py-4 rounded-3xl border-2 backdrop-blur-xl shadow-2xl flex items-center gap-4 ${ROI_STEPS[step].color.split(' ')[0]} bg-slate-950/90`}>
                            <div className="bg-white/10 p-2 rounded-full"><Info className="w-4 h-4" /></div>
                            <div>
                              <p className="text-white font-black text-sm">{ROI_STEPS[step].label}</p>
                              <p className="text-xs text-slate-400">Drag a box over this target</p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="px-8 py-6 bg-slate-900/80 border-t border-slate-800 flex justify-between items-center">
                      <div className="text-sm font-medium text-slate-400">
                        {step < ROI_STEPS.length ? "Define regions to calibrate analysis..." : "Ready to analyze spectral data"}
                      </div>
                      {step === ROI_STEPS.length && !results && (
                        <button 
                          onClick={analyzeColors} 
                          disabled={isAnalyzing} 
                          className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white px-10 py-4 rounded-2xl font-black transition-all flex items-center gap-3 shadow-2xl shadow-blue-600/30 active:scale-95"
                        >
                          {isAnalyzing ? <Zap className="w-6 h-6 animate-spin" /> : <Zap className="w-6 h-6" />}
                          {isAnalyzing ? 'Processing Spectral Data...' : 'Run Vision AI Analysis'}
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Right Column: Workflow & Results */}
              <div className="lg:col-span-4 space-y-8">
                <div className="bg-slate-900 rounded-[32px] border border-slate-800 p-8 shadow-xl">
                  <h3 className="text-xl font-bold mb-8 flex items-center gap-3">
                    <Pipette className="w-6 h-6 text-blue-500" />
                    Calibration Path
                  </h3>
                  <div className="space-y-4">
                    {ROI_STEPS.map((s, idx) => (
                      <div key={String(s.key)} className={`flex items-center justify-between p-4 rounded-2xl border-2 transition-all duration-300 ${rois[s.key] ? 'bg-emerald-500/5 border-emerald-500/20' : step === idx ? 'bg-blue-600/5 border-blue-600/40' : 'border-slate-800/50 opacity-40'}`}>
                        <div className="flex items-center gap-4">
                          <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-black ${rois[s.key] ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/20' : step === idx ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'bg-slate-800 text-slate-500'}`}>
                            {idx + 1}
                          </div>
                          <div>
                            <span className={`block font-bold text-sm ${step === idx ? 'text-white' : 'text-slate-400'}`}>{String(s.label)}</span>
                            {rois[s.key] && <span className="text-[10px] text-emerald-500 font-bold uppercase tracking-widest">Locked</span>}
                          </div>
                        </div>
                        {rois[s.key] && <CheckCircle2 className="w-6 h-6 text-emerald-500" />}
                      </div>
                    ))}
                  </div>
                </div>

                {results && (
                  <div className="animate-in fade-in slide-in-from-right-4 duration-700 space-y-6">
                    <div className="bg-slate-900 rounded-[32px] border border-slate-800 overflow-hidden shadow-2xl">
                      <div className="p-8 bg-gradient-to-br from-slate-900 to-slate-950">
                        <div className="flex items-center justify-between mb-8">
                          <h3 className="text-2xl font-black text-white">Spectral Results</h3>
                          <div className={`px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-[0.2em] shadow-lg ${results.winner === 'A' ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-blue-500/10' : 'bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 shadow-emerald-500/10'}`}>
                            Winner: {results.winner}
                          </div>
                        </div>

                        <div className="space-y-8 mb-10">
                          <div>
                            <div className="flex justify-between text-sm mb-3">
                              <span className="text-slate-400 font-bold uppercase tracking-wider text-[11px]">Dominance: A</span>
                              <span className="text-white font-black text-lg">{String(results.pctA.toFixed(1))}%</span>
                            </div>
                            <div className="h-4 bg-slate-800 rounded-full overflow-hidden p-0.5">
                              <div className="h-full bg-blue-600 rounded-full transition-all duration-[2000ms] shadow-[0_0_20px_rgba(37,99,235,0.4)]" style={{ width: `${results.pctA}%` }} />
                            </div>
                            <div className="mt-2 text-[10px] text-slate-600 font-bold uppercase tracking-widest">DeltaE: {String(results.dA.toFixed(2))}</div>
                          </div>

                          <div>
                            <div className="flex justify-between text-sm mb-3">
                              <span className="text-slate-400 font-bold uppercase tracking-wider text-[11px]">Dominance: B</span>
                              <span className="text-white font-black text-lg">{String(results.pctB.toFixed(1))}%</span>
                            </div>
                            <div className="h-4 bg-slate-800 rounded-full overflow-hidden p-0.5">
                              <div className="h-full bg-emerald-600 rounded-full transition-all duration-[2000ms] shadow-[0_0_20px_rgba(5,150,105,0.4)]" style={{ width: `${results.pctB}%` }} />
                            </div>
                            <div className="mt-2 text-[10px] text-slate-600 font-bold uppercase tracking-widest">DeltaE: {String(results.dB.toFixed(2))}</div>
                          </div>
                        </div>

                        <div className="flex gap-4 mb-10">
                          <button 
                            onClick={copyResultsToClipboard}
                            className="flex-grow bg-slate-800 hover:bg-slate-700 text-white px-4 py-3 rounded-2xl font-bold text-sm transition-all flex items-center justify-center gap-2 border border-slate-700"
                          >
                            {copySuccess ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                            {copySuccess ? "Copied" : "Copy Summary"}
                          </button>
                          <button 
                            className="bg-slate-800 hover:bg-slate-700 text-white p-3 rounded-2xl transition-all border border-slate-700"
                            onClick={() => window.print()}
                          >
                            <Download className="w-5 h-5" />
                          </button>
                        </div>

                        <div className="bg-slate-950/50 border border-slate-800/50 p-6 rounded-3xl relative overflow-hidden group">
                          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                            <Microscope className="w-20 h-20" />
                          </div>
                          <div className="flex items-center gap-3 mb-4">
                            <Info className="w-4 h-4 text-blue-500" />
                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Technical Audit</span>
                          </div>
                          <div className="prose prose-invert prose-sm relative z-10">
                            {results.report.split('\n').map((para, i) => (
                              <p key={i} className="text-slate-400 leading-relaxed text-[13px] mb-4 last:mb-0 font-medium italic italic">
                                {String(para)}
                              </p>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Technical Footer */}
      <footer id="about" className="bg-slate-900/30 border-t border-slate-900 py-20 px-6 mt-20">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-12">
          <div className="md:col-span-2">
            <div className="flex items-center gap-3 mb-6">
              <Crosshair className="w-8 h-8 text-blue-500" />
              <h5 className="text-2xl font-black">Color Examiner AI</h5>
            </div>
            <p className="text-slate-500 text-sm max-w-md leading-relaxed mb-8">
              A professional spectral analysis tool developed for rapid, non-destructive chemical test strip comparison. 
              Our vision engine operates in the CIE L*a*b* color space for perceptual uniformity.
            </p>
            <div className="flex gap-4">
              <a href="#" className="p-2 bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"><Share2 className="w-5 h-5" /></a>
              <a href="#" className="p-2 bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"><ExternalLink className="w-5 h-5" /></a>
            </div>
          </div>
          
          <div>
            <h6 className="font-bold text-white mb-6 uppercase tracking-widest text-xs">Algorithms</h6>
            <ul className="space-y-4 text-sm text-slate-500">
              <li>CIE L*a*b* Perceptual Space</li>
              <li>DeltaE Euclidean Distance</li>
              <li>Von Kries WB Scaling</li>
              <li>Median Filtering for Noise</li>
            </ul>
          </div>

          <div>
            <h6 className="font-bold text-white mb-6 uppercase tracking-widest text-xs">Platform</h6>
            <ul className="space-y-4 text-sm text-slate-500">
              <li>Google Gemini 3 Pro</li>
              <li>React 19 Frontend</li>
              <li>Tailwind CSS Design</li>
              <li>Lucide Vector Graphics</li>
            </ul>
          </div>
        </div>
        <div className="max-w-7xl mx-auto border-t border-slate-800/50 mt-16 pt-8 flex justify-between items-center text-[10px] font-bold text-slate-600 uppercase tracking-[0.2em]">
          <span>© 2025 Visionary Chemical Systems</span>
          <span>Experimental Clinical Prototyping</span>
        </div>
      </footer>

      {error && (
        <div className="fixed bottom-8 right-8 bg-red-600 text-white p-5 rounded-[24px] flex items-center gap-4 backdrop-blur-3xl shadow-2xl animate-in slide-in-from-right-8 max-w-sm z-[100] border border-red-500">
          <div className="bg-white/20 p-2 rounded-full"><AlertCircle className="w-6 h-6" /></div>
          <div>
            <p className="font-black text-sm uppercase tracking-wider">System Error</p>
            <p className="text-xs opacity-80">{String(error)}</p>
          </div>
          <button onClick={() => setError(null)} className="ml-2 hover:scale-125 transition-transform"><X className="w-5 h-5" /></button>
        </div>
      )}

      <style>{`
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #020617; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #334155; }
        html { scroll-behavior: smooth; }
        @media print {
          nav, footer, .lg:col-span-8, button { display: none !important; }
          .lg:col-span-4 { width: 100% !important; border: none !important; }
          body { background: white !important; color: black !important; }
          .text-white { color: black !important; }
          .bg-slate-900 { background: white !important; }
        }
      `}</style>
    </div>
  );
}

const root = createRoot(document.getElementById('root')!);
root.render(<ColorExaminer />);
