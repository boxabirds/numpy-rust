# numpy-rust WebGPU Demo

Interactive demo showcasing GPU-accelerated operations in the browser.

## Requirements

- **Browser:** Chrome 113+ or Edge 113+ (WebGPU support required)
- **GPU:** WebGPU-compatible graphics card with updated drivers
- **Runtime:** Bun 1.0+ (for development)

## Quick Start

```bash
# From repository root
cd demo

# Install dependencies
bun install

# Build WASM module with GPU support
bun run build:wasm

# Start development server
bun run dev
```

Visit `http://localhost:5173` to see the demo.

## Features

- 🚀 WebGPU detection and availability check
- 🎮 GPU information display
- 📊 Matrix multiplication benchmarking (coming soon)
- 🧮 Element-wise operations demo (coming soon)

## Browser Compatibility

| Browser | Version | WebGPU Support |
|---------|---------|---------------|
| Chrome  | 113+    | ✅ Yes         |
| Edge    | 113+    | ✅ Yes         |
| Firefox | 126+    | ⚠️ Experimental |
| Safari  | TP      | 🚧 In Progress  |

### Enabling WebGPU

If WebGPU is not available:

1. Visit `chrome://flags/#enable-unsafe-webgpu`
2. Enable "Unsafe WebGPU"
3. Restart browser

## Building for Production

```bash
cd demo
bun run build

# Deploy dist/ folder to static hosting
```

## License

Apache 2.0
