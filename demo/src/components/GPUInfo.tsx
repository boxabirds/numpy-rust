import { Cpu, AlertCircle } from 'lucide-react';
import { useState, useEffect } from 'react';

interface Props {
  wasmModule: any;
}

interface GpuInfo {
  name: string;
  backend: string;
  vendor: string;
  device: string;
}

export default function GPUInfo({ wasmModule }: Props) {
  const [gpuAvailable, setGpuAvailable] = useState<boolean | null>(null);
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    checkGpu();
  }, [wasmModule]);

  async function checkGpu() {
    try {
      const available = wasmModule.is_gpu_available();
      setGpuAvailable(available);

      if (available) {
        const info = wasmModule.get_gpu_info();
        setGpuInfo(info);
      }
    } catch (err) {
      setError(`Failed to check GPU: ${err}`);
      setGpuAvailable(false);
    }
  }

  if (error) {
    return (
      <div className="gpu-info error">
        <AlertCircle size={20} />
        <span>{error}</span>
      </div>
    );
  }

  if (gpuAvailable === null) {
    return (
      <div className="gpu-info loading">
        <Cpu size={20} />
        <span>Checking GPU...</span>
      </div>
    );
  }

  if (!gpuAvailable) {
    return (
      <div className="gpu-info unavailable">
        <AlertCircle size={20} />
        <div className="gpu-info-text">
          <span className="gpu-status">GPU Not Available</span>
          <span className="gpu-detail">CPU-only mode</span>
        </div>
      </div>
    );
  }

  return (
    <div className="gpu-info available">
      <Cpu size={20} />
      <div className="gpu-info-text">
        <span className="gpu-status">GPU Enabled</span>
        {gpuInfo && (
          <span className="gpu-detail">
            {gpuInfo.name} ({gpuInfo.backend})
          </span>
        )}
      </div>
    </div>
  );
}
