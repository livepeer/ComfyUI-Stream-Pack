# ComfyUI Node Pack

A suite of custom **ComfyUI nodes** for building **real-time video and audio workflows** using [ComfyStream](https://github.com/yondonfu/comfystream). This pack introduces nodes that enable **live streaming, AI-enhanced processing, and real-time interaction** with ComfyUI workflows.

## Node Categories

The node pack is organized into three categories based on their function:  

### Foundation Nodes

Foundation Nodes introduce **entirely new capabilities** to ComfyUI, unlocking workflows that were previously impossible. These nodes typically:

- Introduce new **AI models** or **stable diffusion checkpoints**.
- Enable **core functionality** that can be further expanded with Light Nodes.  

> [!NOTE]
> No Foundation Nodes have been added yet.  

### Light Nodes

Light Nodes **modify or enhance existing workflows** by:  

- **Tweaking parameters** or intermediate values in Foundation Nodes.  
- Providing **workflow insights** or debug information.  
- Creating **variations** of existing functionalities for new use cases.  

> [!NOTE]
> No Light Nodes have been added yet.

### Input/Output Nodes

These nodes **extend ComfyUI's input and output capabilities**, making real-time streaming workflows possible. They allow users to:

- Load and process **video, audio, or tensor data** from external sources.  
- Stream outputs to **Livepeer, WebRTC, or other real-time platforms**.  
- Connect **external data sources** for AI-driven workflows.  

Examples:

- [**LoadTensor**](https://github.com/yondonfu/comfystream/blob/main/nodes/tensor_utils/load_tensor.py) - Loads tensor data for real-time processing.  
- [**LoadAudioTensor**](https://github.com/yondonfu/comfystream/blob/main/nodes/audio_utils/load_audio_tensor.py) - Processes real-time audio input.  

> [!NOTE]
> No additional Input/Output Nodes have been added yet.
