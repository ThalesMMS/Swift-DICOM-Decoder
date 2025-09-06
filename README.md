# Swift DICOM Decoder

A Swift port of the DICOM decoder originally created in Objective-C by Luo Zhaohui (kesalin). This version reimplements the core functionality in modern Swift, taking advantage of current frameworks for improved performance and safety.

The codebase has been rewritten in Swift with automatic memory management through ARC, type-safe data structures, and GPU acceleration via Metal. CPU-intensive operations use the Accelerate framework for vectorized calculations.

**Port Author:** Thales Matheus

**Original Objective-C Project:** [kesalin/DicomDecoder](https://github.com/kesalin/DicomViewer) (by Luo Zhaohui)

* * *

## Project Overview

A library for parsing, processing, and rendering DICOM medical images. Works with standard DICOM files and can be integrated into medical imaging applications on Apple platforms.

### Core Features

- **DICOM Parsing:** Reads headers, metadata tags, and pixel data from .dcm files
- **Image Format Support:** Handles 8-bit and 16-bit grayscale plus 24-bit RGB color images
- **2D Rendering:** Custom UIView implementation with aspect ratio preservation
- **GPU Windowing:** Window/level adjustments using Metal compute shaders for real-time performance
- **CPU Fallbacks:** GCD-based parallel processing with Accelerate framework when Metal unavailable
- **Image Utilities:** Windowing presets, thumbnail generation, and quality metrics
    

* * *

## Original vs. Swift Port Comparison

The Swift port includes architectural improvements beyond syntax translation:

Feature Area

Original Objective-C Implementation

Modern Swift Port

**Why It Matters**

**Language & Safety**

Objective-C / Objective-C++

**Modern Swift (5+)**

Provides compile-time type safety, optionals to prevent null-pointer errors, and a cleaner, more expressive syntax.

**Memory Management**

Manual (`alloc`, `release`, `autorelease`)

**Automatic Reference Counting (ARC)**

Eliminates the cognitive overhead of manual memory management, preventing common leaks and memory corruption bugs.

**Data Handling**

Raw Pointers (`Byte *`, `ushort *`), `NSData`

`Data`, `[UInt8]`, `[UInt16]`, `withUnsafeBytes`

Guarantees memory safety, prevents buffer overflows, and provides a safer, more intuitive API for handling pixel data.

**Windowing Performance**

CPU-bound, pixel-by-pixel loop

**1\. Metal GPU Shaders**

**2\. Parallel CPU (GCD)**

**3\. Vectorized CPU (Accelerate)**

Significant performance improvements through GPU offloading. CPU remains free for other tasks with fallback options ensuring consistent performance across hardware.

**Code Architecture**

`NSObject`\-based classes

**`structs` for pure logic**, `final classes` for objects, and type-safe `enums`

Promotes value semantics for better performance and predictability. A clear separation of concerns makes the code more testable and maintainable.

**Error Handling**

`BOOL` flags (`dicomFileReadSuccess`)

**`Result` type** and a custom `DicomToolError` enum

Makes error paths explicit and robust. Failures are handled gracefully and provide clear, descriptive reasons, which is critical for debugging.

**I/O Operations**

Standard `NSData` file reading

**Memory-mapped files** for large datasets

Drastically reduces initial load times and memory footprint for large DICOM series by avoiding reading the entire file into RAM at once.

**Extensibility**

Core functionality only

**\- Optimized Thumbnails**

**\- Batch Processing**

**\- Quality Metrics**

**\- Async/Await API**

The new architecture is modular and easily extensible, allowing for the addition of powerful new features.

Exportar para as Planilhas

* * *

## Architecture

The project is organized into specialized components:

### DCMDecoder.swift

Parses binary DICOM files.

- **Responsibilities:** Reads .dcm files, parses tags, extracts metadata and pixel data
- **Key Features:**
  - Typed enums for DICOM tags and Value Representations
  - Memory-mapped I/O for files over 10MB using `Data(contentsOf:options: .mappedIfSafe)`
  - Compressed syntax support via ImageIO for JPEG/JPEG2000 formats
        

### DCMImgView.swift

UIView subclass for high-performance rendering.

- **Responsibilities:** Renders pixel data and handles window/level updates
- **Rendering Pipeline:**
  1. Receives 16-bit pixel buffer from DicomTool
  2. Dispatches Metal compute shader for window/level transformation
  3. GPU processes pixels in parallel to 8-bit output
  4. Creates CGImage from processed buffer
  5. Draws final image in view's draw method
- **Fallback:** CPU-based processing using DispatchQueue.concurrentPerform when Metal unavailable
        

### DCMWindowingProcessor.swift

Stateless struct for image processing algorithms.

- **Responsibilities:** Window/level calculations, histogram generation, image quality analysis
- **Features:**
  - Decoupled algorithms for testability and reusability
  - Accelerate framework's vDSP for vectorized operations using SIMD instructions
        

### DicomTool.swift

Orchestrator between decoder and view layers.

- **Responsibilities:** Manages workflow, decoder initialization, pixel data transfer, window/level state
- **Features:**
  - @MainActor for thread-safe UI updates
  - async/await interface for modern Swift concurrency
  - Result type with DicomToolError for clear error handling
        

* * *

## Usage

Integration example for iOS or macOS applications:

```Swift

    import UIKit
    
    class DicomViewController: UIViewController {
    
        private var dicomDecoder: DCMDecoder!
        private var dicomView: DCMImgView!
        private var dicomTool: DicomTool!
    
        override func viewDidLoad() {
            super.viewDidLoad()
            setupDicomViewer()
            loadImage()
        }
    
        private func setupDicomViewer() {
            // 1. Initialize the core components
            dicomDecoder = DCMDecoder()
            dicomView = DCMImgView(frame: self.view.bounds)
            dicomTool = DicomTool.shared
            
            // Configure the view
            dicomView.contentMode = .scaleAspectFit
            self.view.addSubview(dicomView)
        }
    
        private func loadImage() {
            // 2. Get the path to your DICOM file
            guard let dicomPath = Bundle.main.path(forResource: "sample_image", ofType: "dcm") else {
                print("Error: DICOM file not found in bundle.")
                return
            }
            
            // 3. Use the DicomTool to orchestrate decoding and display
            // This single call handles file loading, parsing, and initial rendering.
            let result = dicomTool.decodeAndDisplay(
                path: dicomPath,
                decoder: dicomDecoder,
                view: dicomView
            )
            
            // 4. Handle the result
            switch result {
            case .success:
                print("DICOM image loaded and displayed successfully!")
                // You can now access metadata from the decoder
                let patientName = dicomDecoder.info(for: 0x00100010) // Patient's Name Tag
                print("Patient Name: \(patientName)")
                
            case .failure(let error):
                print("Failed to process DICOM file: \(error.localizedDescription)")
            }
        }
        
        // You can later adjust the windowing interactively
        func updateUserWindowLevel(newWidth: Int, newCenter: Int) {
            dicomTool.applyWindowLevel(windowWidth: newWidth, windowCenter: newCenter, view: dicomView)
        }
    }
```

* * *

## Limitations

This decoder currently does not support decompression of compressed DICOM images (like JPEG Baseline, JPEG Extended, JPEG Lossless, JPEG 2000, JPEG-LS and RLE.

The decoder works only with uncompressed DICOM files using the following transfer syntaxes:
- Implicit VR Little Endian (1.2.840.10008.1.2)
- Explicit VR Little Endian (1.2.840.10008.1.2.1)
- Explicit VR Big Endian (1.2.840.10008.1.2.2)

For compressed DICOM files, consider using external tools to decompress them first, such as DCMTK's `dcmdjpeg` utility.

* * *

## Future Development

Potential enhancements:

- Multi-frame support for DICOM cine loops
- Additional transfer syntax support (JPEG-LS, RLE)
- Interactive UI controls (windowing, pan, zoom, annotations)
- DICOM networking (C-STORE, C-FIND, C-GET/C-MOVE)
- Swift Package Manager distribution
    

* * *

## License

The original Objective-C code by Luo Zhaohui (kesalin) did not include an explicit license. This Swift port is available under the MIT License.

The MIT License permits use, copy, modification, merge, publish, and distribution for any purpose, provided the original copyright notice and permission notice are included.

Please provide attribution to both the original author, Luo Zhaohui, and the port author, Thales Matheus, in derivative works.

Thanks to Luo Zhaohui for the original implementation that served as the basis for this Swift version.