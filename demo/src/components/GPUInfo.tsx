import { Cpu, AlertCircle } from 'lucide-react';

interface Props {
  gpuAvailable: boolean;
}

export default function GPUInfo({ gpuAvailable }: Props) {
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
        <span className="gpu-detail">WebGPU ready</span>
      </div>
    </div>
  );
}
